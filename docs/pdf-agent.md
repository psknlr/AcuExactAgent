# PDF 智能体与 API 接入

启动 `python -m acuexact serve --port 8765`，打开 `http://127.0.0.1:8765`。界面主入口为导入 PDF、填写 prompt、配置模型、运行。运行结束可查看每条古籍引文的强弱相关标签、维度、理由、页码、原文明示书名/篇章和待审事项，并导出 Excel 可用的 TSV 与完整 JSON。

## 供应商

| 选择 | 协议 | 默认模型示例 | 图片输入 | 环境变量 |
| --- | --- | --- | --- | --- |
| OpenAI / GPT | Responses + 严格 JSON Schema | `gpt-4.1` | 开启视觉后可用，须模型本身支持 | `OPENAI_API_KEY` |
| Anthropic / Claude | Messages + `output_config.format` | `claude-sonnet-4-6` | 开启视觉后可用，须模型本身支持 | `ANTHROPIC_API_KEY` |
| MiniMax | OpenAI-compatible Chat Completions + 本地 JSON 校验 | `MiniMax-M2.5` | 本适配器只用于文字；扫描件配辅助视觉模型 | `MINIMAX_API_KEY` |
| 自定义 OpenAI-compatible | Chat Completions + 本地 JSON 校验 | 用户填写 | 用户确认模型支持并开启 | 显式表单 key 或 CLI 指定环境变量 |

模型名可编辑，示例不是账户模型可用性承诺。OpenAI/Claude 模型须支持对应结构化输出协议；不支持时会明确返回错误。MiniMax 支持国际 `https://api.minimax.io/v1`、中国 `https://api.minimaxi.com/v1` 和 `https://api.minimax.cn/v1`。自定义服务须 HTTPS，不能只把 Anthropic 原生 Messages 地址填成 OpenAI-compatible。

“测试连接”才发出小型真实 API 请求，可能消耗 token；加载页面和切换供应商不调用模型。运行 PDF 智能体会将所选页的文字、版面信息及启用视觉时的图片发送到选定供应商。PDF 上传先保存在本机临时目录。API Key 不写入文件、浏览器存储、结果或日志；可留空使用服务进程的对应环境变量。关闭页面不会自动停止后台任务，使用取消按钮；停止本地服务清理临时文档。生产部署需另行配置认证和资源隔离，本服务只接受 localhost/127.0.0.1 访问。

## 可复现命令

PowerShell 配置主模型后运行（将占位内容替换为自己的凭据与文件）：

```powershell
$env:MINIMAX_API_KEY = "YOUR_KEY"
python -m acuexact agent --pdf paper.pdf --provider minimax --model MiniMax-M2.5 --prompt-file examples/research-prompt.txt --title "EndNote 中的文章标题" --out outputs/minimax
```

扫描件由 Claude 读图，MiniMax 负责研究分类：

```powershell
$env:ANTHROPIC_API_KEY = "YOUR_KEY"
python -m acuexact agent --pdf paper.pdf --provider minimax --model MiniMax-M2.5 --vision-provider anthropic --vision-model claude-sonnet-4-6 --prompt-file examples/research-prompt.txt --pages 1-5 --out outputs/scan
```

GPT 同时负责文字与视觉：

```bash
python -m acuexact agent --pdf paper.pdf --provider openai --model gpt-4.1 --vision --prompt-file examples/research-prompt.txt --out outputs/gpt
```

自定义网关使用 `--provider openai_compatible --base-url https://YOUR_HOST/v1 --model MODEL --api-key-env CUSTOM_API_KEY`；它不会继承 `OPENAI_API_KEY`。不要把 API Key 作为 CLI 参数或写进 prompt。Linux/macOS 使用 `export MINIMAX_API_KEY='YOUR_KEY'`。Docker 可用 `--env-file` 加载本地环境文件，或在浏览器表单输入 Key；不要把环境文件提交 Git。

## 执行与质量约束

1. 本地读取 PDF 和文字质量。扫描、乱码、引号异常会触发视觉读取或明确待审。每张图片限制长边约 1900 px，转录保留原文字层和不确定项。
2. 模型根据标题、页码、文本坐标、分隔线和视觉版面规划一篇文章的有序选区。存在多篇而未选标题，返回候选标题要求选择，不默认混合整份 PDF。选区必须在页面内且互不重叠。
3. 本地扫描全部成对中文引号，仅排除精确病名；模型查漏候选必须在原始选区逐字对齐。视觉转录始终标记待复核，不能当作原 PDF 字符的精确金标准。
4. 按输入 prompt 每批最多 20 条分类，检查 ID 不漏不重、标签和维度有效、理由非空；失败允许一次修复，仍失败则保留原文并显示“待分类”。
5. 保存 PDF SHA256、原 prompt、选区、页码、API 请求元数据、调用预算与阶段记录。主/视觉模型共享 `max_calls` HTTP 请求预算，429/5xx重试和JSON修复均计入。只展示步骤摘要，不展示模型内部思维链。取消和预算耗尽保留已取得结果。

网页单文件上限 40 MB / 200 页，默认每次分析最多 30 页，支持 `1-3,5` 页码范围；超限明确报错，不静默截断。视觉规划单次最多20页，规划输入JSON最多180,000字符，超过需缩小页码范围。并行执行 2 个任务，最多 4 个任务排队或运行；服务本次最多保留 10 份 PDF、最近 50 个任务。长文建议按目标文章分批。

视觉审阅与文字层不一致时，两种文字均作为带来源的待审记录保留，可能形成需要人工合并的双证据条目；不能把它们未经复核当作两个独立原文引文。视觉条目的 `offset_space` 指向具体区域的转录文本，不能当作 PDF 文字层偏移。

原文明示出处永远不凭模型常识补全。智能体默认不联网考证出处；有需要可对输出另运行 `python -m acuexact lookup --input outputs/gpt/result.json --out outputs/looked-up`。候选来源与原文明示分栏，核证前保留待查。

## API 与验证边界

`POST /api/uploads` 接收原始 `application/pdf`，`X-Filename` 为 URI 编码的文件名。返回临时 document_id。`POST /api/jobs` 接收 document_id、prompt、target_title、pages、config 与可选 vision_config；返回 job_id。`GET /api/jobs/{id}` 获取状态、阶段事件和 result。`POST /api/jobs/{id}/cancel` 取消后续步骤。`GET /api/documents/{id}/pages/{page}.png` 预览原页。`GET /api/providers` 返回可用适配器与环境变量是否已设置；不会返回密钥。

已经用模拟 API 验证各供应商请求协议、结构化响应、重试与失败路径；没有使用用户真实 API Key 做远端模型效果测试。扫描识别、自动分篇、强弱分类均可能出错，需要原页复核。原压缩包缺少真实人工 gold，因此没有实际准确率承诺。

官方协议依据：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[Claude Vision](https://platform.claude.com/docs/en/build-with-claude/vision)、[MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api)。
