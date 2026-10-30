# 架构与数据约定

## 证据优先的 Agent 阶段

| 阶段 | 模块 | 约束 |
| --- | --- | --- |
| 资料清点 | ingest | 不全量解压；路径、大小、重复成员检查 |
| 目标规划 | planner | EndNote ID 稳定；文件候选不能自行批准文章范围 |
| 版面读取 | pdf | 页面渲染、文字质量检测、备用解析器、PDF 哈希绑定、已审区域 |
| 原文提取 | extraction | 平衡引号扫描；不因强弱相关筛选；不可猜字或补源 |
| 研究分类 | classification | 离线规则或可选结构化模型；原文和显式出处只读 |
| 来源检索 | retrieval | 明确启用联网；正文精确匹配；检索命中只是候选 |
| 核证补全 | classification.enrich_source | 仅采纳带审核人与证据的 catalog；冲突保留待审 |
| 评测导出 | evaluation / export | gold 不进入推理；逐条多重集匹配；TSV 公式防护 |
| PDF 智能体 | agent | 模型规划选区、视觉读取、查漏与限次修复、逐批分类、不可变证据校验 |
| API 适配 | providers | OpenAI Responses / Claude Messages / MiniMax / 自定义；结构与错误校验、有限重试 |
| 本地任务 | workspace / server | 临时 PDF、限量队列、异步进度、取消、原页预览；密钥不持久化 |

这是可审计的阶段式 Agent 基线，不是经过真实金标准训练的模型，也不声称具备无需人工复核的万能 OCR、分篇或古籍考证能力。

## Quote schema（主要字段）

```json
{
  "id": "endnote-8:q0001",
  "article_id": "endnote-8",
  "text": "肩背疼痛，循经取穴。",
  "text_raw": "肩背疼痛，\n循经取穴。",
  "char_start": 12,
  "char_end": 29,
  "source_explicit": {"book": null, "chapter": null, "evidence": null},
  "source_enriched": {"book": null, "chapter": null, "evidence": null, "url": null, "status": "待查证"},
  "evidence": [{"page": 1, "bbox": [40, 80, 290, 700], "region_id": "p1-left", "method": "pdf_text", "text": "“肩背疼痛，"}],
  "relevance": {"label": "强相关", "dimensions": ["肩痛取穴"], "reason": "...", "method": "rules", "needs_review": true},
  "review_flags": ["source_not_explicit"]
}
```

此处数字仅示意，实际偏移来自程序。字符偏移采用 Python Unicode 字符索引，指向按清单顺序以换行连接的区域文本；不是字节偏移或 PDF 字形编号。`char_end` 为不包含的结束位置，外层引号计入范围。引用跨区域时会有多条 evidence。

缺出处不是丢弃条件，也不等于书籍不存在。`null`、待查和已核证有不同含义。

## 核证 catalog

```json
[
  {
    "book": "演示医籍",
    "chapter": "演示篇",
    "evidence": "这是合成核证样例：肩背疼痛，循经取穴。",
    "url": "https://example.org/reviewed-source",
    "verified": true,
    "reviewer": "实际核证人"
  }
]
```

示例域名不可作为真实出处。真实条目须有人核实书名、篇章、版本和引文；`verified` 是已完成审核的声明，不是程序凭网页出现了字符串自动生成的事实。程序要求证据包含完整引文（仅忽略空白）、URL 有效、审核人非空。未获核证、多来源冲突或与原文明示书名不一致，不自动补全。即使已核证也永远写在 `source_enriched`，不覆盖 `source_explicit`。

## 可选模型

新版 `agent` 的模型接口详见 [PDF 智能体](pdf-agent.md)。OpenAI 使用 `text.format`，Claude 使用 `output_config.format`；MiniMax 和自定义兼容接口使用提示词 JSON 与本地 jsonschema 校验，不能等同于原生严格输出。兼容接口格式不合格最多修复一次；429/5xx 最多额外重试两次，超时不重发，HTTP 重定向不转发密钥。主/视觉模型共享请求预算，修复和重试也计入。

旧 `extract --llm` 仍提供 OpenAI 分类接口。所有路径均把 PDF/引文当待分析数据，模型分类只返回 ID、label、dimensions、reason、needs_review，不能更改原文或补出处。不完整响应、拒绝、非法维度、缺少/重复/多出的 ID 和没有维度的“强相关”均拒绝接受。OpenAI 设置 `store:false`。自定义 API 须使用明确选择的 HTTPS 兼容端点和独立凭据；官方供应商固定官方主机。错误不记录凭据或响应正文。

参考：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)。模型性能和供应商存储策略需以实际账户、模型及其文档为准；此项目仅显式设置不保存 Responses 对象。

## 检索边界

默认不联网。用户启用 `lookup` 后固定访问中文维基文库 API，一次搜索加最多两页正文；每请求默认 10 秒，上限 30 秒，响应最多 4 MiB。不会因搜索摘要命中就认定来源，不跟随任意重定向，记录页 ID/版本 ID/精确证据。汇编和转引页面可能包含相同句子，因此命中结果仍是候选，不冒充古籍最早出处。

参考：[MediaWiki Search](https://www.mediawiki.org/wiki/API:Search)、[MediaWiki Parsing](https://www.mediawiki.org/wiki/API:Parsing_wikitext)。简繁、异体、标点和版本差异会影响精确匹配召回；当前不自动改写这些差异。

## 评测约定

按 `(article_id, text)` 对每次出现做独立匹配；同文章重复句不合并，gold 必须保持出现顺序。逐字主指标不接受近似命中；忽略全部空白的辅助指标单列。漏提、串文和多提都会反映为 FN/FP。相似字符串仅供人工定位错字。

gold JSON 可加入 `relevance.label` 和 `relevance.dimensions`。分类指标只在成功匹配且有人工标签的引文上计算，因此须同时看提取 recall。没有人工标签的字段返回不可用，不记作零错误。建议另外统计边界错误、OCR错误、出处归属错误、标注歧义和待审率。

源文、gold、配置和模型版本应各自留哈希；报告留存 Git commit、模型 ID、prompt_version 及 API response_id。现有实现保存 PDF/text 哈希、配置审计和模型响应标识；真实数据集分组哈希与人工审核记录由项目研究者管理。
