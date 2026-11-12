# PDF 智能体与 API 接入

启动 `python -m acuexact serve --port 8765`，打开 `http://127.0.0.1:8765`。界面主入口为导入 PDF、填写 prompt、配置模型、运行。运行结束可查看每条古籍引文的强弱相关标签、维度、理由、页码、原文明示书名/篇章和待审事项，并导出 Excel 可用的 TSV 与完整 JSON。

## 供应商

| 选择 | 协议 | 默认模型示例 | 图片输入 | 环境变量 |
| --- | --- | --- | --- | --- |
| OpenAI / GPT | Responses + 严格 JSON Schema | `gpt-4.1` | 开启视觉后可用，须模型本身支持 | `OPENAI_API_KEY` |
| Anthropic / Claude | Messages + `output_config.format` | `claude-sonnet-4-6` | 开启视觉后可用，须模型本身支持 | `ANTHROPIC_API_KEY` |
| MiniMax | OpenAI-compatible Chat Completions + 本地 JSON 校验 | `MiniMax-M3.1-Flash` | 开启视觉后支持图片；M2 系列仍仅支持文字 | `MINIMAX_API_KEY` |
| 自定义 OpenAI-compatible | Chat Completions + 本地 JSON 校验 | 用户填写 | 用户确认模型支持并开启 | 显式表单 key 或 CLI 指定环境变量 |

模型名可编辑，示例不是账户模型可用性承诺。OpenAI/Claude 模型须支持对应结构化输出协议；不支持时会明确返回错误。MiniMax **默认使用国内站 `https://api.minimax.cn/v1`**，与当前[国内官方文档](https://platform.minimax.cn/docs/api-reference/text-openai-api)一致；界面的“MiniMax 服务站点”可切换国内原域名 `https://api.minimaxi.com/v1` 或国际站 `https://api.minimax.io/v1`，也可直接编辑 Base URL。使用所选站点对应的 API Key，切换站点会清空手动填写的 Key；后端 `MINIMAX_API_KEY` 须由使用者配置为对应站点凭据。请求失败不会自动切换域名或站点。

MiniMax 通过 `POST {Base URL}/chat/completions` 与 `Authorization: Bearer <API Key>` 调用，不要把 `/chat/completions` 或 `/anthropic` 填进 Base URL。CLI 不填 `--base-url` 也默认国内站；如使用国内原域名，增加 `--base-url https://api.minimaxi.com/v1`，国际站增加 `--base-url https://api.minimax.io/v1`。自定义服务须 HTTPS，不能只把 Anthropic 原生 Messages 地址填成 OpenAI-compatible。

“测试连接”才发出小型真实 API 请求，可能消耗 token；加载页面和切换供应商不调用模型。运行 PDF 智能体会将所选页的文字、版面信息及启用视觉时的图片发送到选定供应商。PDF 上传先保存在本机临时目录。API Key 不写入文件、浏览器存储、结果或日志；可留空使用服务进程的对应环境变量。关闭页面不会自动停止后台任务，使用取消按钮；停止本地服务清理临时文档。生产部署需另行配置认证和资源隔离，本服务只接受 localhost/127.0.0.1 访问。

## 可复现命令

PowerShell 配置主模型后运行（将占位内容替换为自己的凭据与文件）：

```powershell
$env:MINIMAX_API_KEY = "YOUR_KEY"
python -m acuexact agent --pdf paper.pdf --provider minimax --model MiniMax-M3.1-Flash --prompt-file examples/research-prompt.txt --title "EndNote 中的文章标题" --out outputs/minimax
```

可选独立视觉模型：以下由 Claude 读图、MiniMax 负责研究分类；`--vision-provider minimax --vision-model MiniMax-M3.1-Flash` 也可独立负责读图：

```powershell
$env:ANTHROPIC_API_KEY = "YOUR_KEY"
python -m acuexact agent --pdf paper.pdf --provider minimax --model MiniMax-M3.1-Flash --vision-provider anthropic --vision-model claude-sonnet-4-6 --prompt-file examples/research-prompt.txt --pages 1-5 --out outputs/scan
```

GPT 同时负责文字与视觉：

```bash
python -m acuexact agent --pdf paper.pdf --provider openai --model gpt-4.1 --vision --prompt-file examples/research-prompt.txt --out outputs/gpt
```

也可明确指定 `gpt-4o-mini`，使用 `OPENAI_API_KEY` 和官方 Responses 接口。该模型上下文为 128,000 token，单次输出上限 16,384 token；高精度图片的 token 用量较高。批处理中启用 `--long-documents` 时，`gpt-4o-mini` 超过两页的 PDF 按两页分段，继续核对文章边界、原始页码和跨段引文。图片保持 `detail=high`，不通过降低读图质量来容纳更多页。账户权限与模型实际可用性仍须用真实请求验证。[模型规格](https://developers.openai.com/api/docs/models/gpt-4o-mini)、[图片输入](https://developers.openai.com/api/docs/guides/images-vision)。

```bash
python -m acuexact agent-batch --source papers.zip --limit 1 --workers 1 --provider openai --model gpt-4o-mini --max-tokens 16000 --long-documents --prompt-file examples/research-prompt.txt --out outputs/gpt4o_mini
```

自定义网关使用 `--provider openai_compatible --base-url https://YOUR_HOST/v1 --model MODEL --api-key-env CUSTOM_API_KEY`；它不会继承 `OPENAI_API_KEY`。不要把 API Key 作为 CLI 参数或写进 prompt。Linux/macOS 使用 `export MINIMAX_API_KEY='YOUR_KEY'`。Docker 可用 `--env-file` 加载本地环境文件，或在浏览器表单输入 Key；不要把环境文件提交 Git。

## 执行与质量约束

1. 本地读取 PDF 和文字质量。扫描、乱码、引号异常会触发视觉读取或明确待审。每张图片最高 180 DPI、限制长边约 1900 px，转录保留原文字层和不确定项。
2. 模型根据标题、页码、文本坐标、分隔线和视觉版面规划一篇文章的有序选区。存在多篇而未选标题，返回候选标题要求选择，不默认混合整份 PDF。选区必须在页面内且互不重叠。
3. 本地扫描全部成对中文引号，仅排除精确病名；模型查漏候选必须在原始选区逐字对齐。视觉转录始终标记待复核，不能当作原 PDF 字符的精确金标准。
4. 按输入 prompt 每批最多 20 条分类，检查 ID 不漏不重、标签和维度有效、理由非空；失败允许一次修复，仍失败则保留原文并显示“待分类”。
5. 保存 PDF SHA256、原 prompt、选区、页码、API 请求元数据、调用预算与阶段记录。主/视觉模型共享 `max_calls` HTTP 请求预算，429/5xx重试和JSON修复均计入。只展示步骤摘要，不展示模型内部思维链。取消和预算耗尽保留已取得结果。

网页单文件上限 40 MB / 200 页，默认每次分析最多 30 页，支持 `1-3,5` 页码范围；超限明确报错，不静默截断。视觉规划单次最多20页，规划输入JSON最多180,000字符，超过需缩小页码范围。并行执行 2 个任务，最多 4 个任务排队或运行；服务本次最多保留 10 份 PDF、最近 50 个任务。长文建议按目标文章分批。

视觉审阅与文字层不一致时，两种文字均作为带来源的待审记录保留，可能形成需要人工合并的双证据条目；不能把它们未经复核当作两个独立原文引文。视觉条目的 `offset_space` 指向具体区域的转录文本，不能当作 PDF 文字层偏移。

原文明示出处永远不凭模型常识补全。智能体默认不联网考证出处；有需要可对输出另运行 `python -m acuexact lookup --input outputs/gpt/result.json --out outputs/looked-up`。候选来源与原文明示分栏，核证前保留待查。

## API 与验证边界

`POST /api/uploads` 接收原始 `application/pdf`，`X-Filename` 为 URI 编码的文件名。返回临时 document_id。`POST /api/jobs` 接收 document_id、prompt、target_title、pages、config 与可选 vision_config；返回 job_id。`GET /api/jobs/{id}` 获取状态、阶段事件和 result。`POST /api/jobs/{id}/cancel` 取消后续步骤。`GET /api/documents/{id}/pages/{page}.png` 预览原页。`GET /api/providers` 返回可用适配器与环境变量是否已设置；不会返回密钥。

已经用模拟 API 验证各供应商请求协议、结构化响应、重试与失败路径。2026-11-03 已用中国内地 MiniMax 接口和 `MiniMax-M3.1-Flash` 跑通一篇真实乱码文字层 PDF 的读图、分区、提取、查漏和分类（10 次请求）。这验证了接口与流程，不构成准确率评测。扫描识别、自动分篇、强弱分类均可能出错，需要原页复核。原压缩包缺少真实人工 gold，因此没有实际准确率承诺。

官方协议依据：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[Claude Vision](https://platform.claude.com/docs/en/build-with-claude/vision)、[MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api)。

## MiniMax M3.1 Flash 视觉与批处理

主模型可直接读图，无需另配 GPT/Claude：

```bash
python -m acuexact agent --pdf paper.pdf --provider minimax --model MiniMax-M3.1-Flash --vision --reasoning-effort low --max-tokens 16000 --timeout 180 --prompt-file examples/research-prompt.txt --title "目标文章标题" --out outputs/minimax
```

MiniMax 使用 `max_completion_tokens`；M3.1 通过 `reasoning_effort` 调整思考深度，支持 `low/medium/high/xhigh/max`，不能关闭 thinking。适配器只解析最终 `content`，不保存 `reasoning_content`。JSON 仍采用提示词约束、本地 schema 校验及一次修复，未声称支持原生严格 JSON。图片通过 `image_url` 发送 base64 data URL，沿用每张 5 MB、最多 20 张的本地限制。

批处理读取 ZIP 或 PDF 文件夹，按目录、文件名中的数字自然排序；默认取前 350 篇。请先用 `--limit 1` 验证目标资料，再扩大范围：

```bash
python -m acuexact agent-batch --source papers.zip --limit 350 --workers 8 --provider minimax --model MiniMax-M3.1-Flash --reasoning-effort low --timeout 180 --prompt-file examples/research-prompt.txt --out outputs/minimax_350
```

- `--start 1` 指定自然排序后的起点；`--prefix "CNKI 1_1187_paper_original/"` 限定文件夹。文件名去掉开头编号作为目标标题，去掉末尾独立数字的下载重名后缀；标题匹配仅规范常见繁简异体、全半角及“针剌/针刺”等书目写法；正文从不应用这些替换。标题不匹配会保留 `needs_scope`，不会自动混入其他文章。
- 独立进程并行，避免 PDF 渲染库跨线程冲突。批量模式直接用页面图像规划，选区文字层清晰时直接保留其原文，损坏或不足时才转录选定区域；短篇和长篇均省去重复的清晰文字转录，使用高细节图片输入。并发可用 `--workers` 调整；请求预算、超时和页数限制均逐篇生效。
- 每篇立即保存到 `article-*/result.json`、`quotes.tsv`、`status.json`、`events.jsonl`。根目录的 `quotes.tsv`、`results.json`、`progress.json` 在每篇结束时原子更新；`manifest.json` 固定文件选择和运行配置。
- TSV 的“文件名（中文）”列保留来源 PDF 的原始文件名（含编号和 `.pdf` 后缀），与 JSON 的 `source_pdf_filename` 对应；使用 ZIP 内的文件名，不使用内部暂存编号或推断的文章标题。续跑旧批次时会补齐这一列，无需重新调用模型。
- 分类请求使用短条号（`q1` 至 `q20`）并在 schema 中枚举允许值，避免模型抄错较长的来源条号；本地保存审计映射，输出继续使用原始条号。重复出现的同文引文保留为独立记录。
- 处理学位论文等长文献时加 `--long-documents`：通常超过 20 页的 PDF 按 10 页一组确认文章与续文边界，`gpt-4o-mini` 使用上述两页分组。完整处理全部页面，`--max-calls` 预算按组生效。每组保存 `parts/part-*/result.json`；有效文字层直接用于规划和提取，损坏文字层仍读图，标题或坐标校验失败时补做图像规划。合并时保留原始页码与区域证据，并补回跨组引文；边界不明或任何一组未完成时不会标记完整结束。旧分组只有来源哈希、prompt、供应商、模型、标题和页码一致时才复用；替换分组前将旧 JSON/TSV 归档到该分组的 `attempts/`，保留实际模型来源。
- 结构异常的 PDF 会在临时副本中规范化交叉引用、页面树或页面尺寸；原始文件与来源 SHA256 保持不变。无法恢复时保留失败状态，不作为零引文结果。
- 同一命令再次运行，会跳过完整结束的文章，重跑未完成项；每篇 `checkpoints/` 保存通过 JSON/schema 校验的阶段结果，复用相同证据的转录。重跑前的 JSON/TSV 另存到 `attempts/`，保留既有证据。语义校验失败的规划和分类不会永久缓存重放。检查点含文献转录文字，仅保存在本地。来源、prompt、模型或配置变化时必须使用新输出目录。密钥只从供应商环境变量读取，不保存到清单或结果。
- 在输出目录创建名为 `STOP` 的文件，停止派发任务并在下次模型调用前取消进行中任务。恢复前删除 `STOP`，重新运行原命令。
- `pipeline_complete=true` 表示提取、查漏和分类流程结束；自动边界与视觉证据仍标为 `needs_review`。`needs_attention` 表示流程未完成，需要检查对应告警；不能把失败当作零引文。处理结束有未完成项时返回非零退出码。
- 请求元数据中的 `status=refused` 单独标记供应商的明确拒绝；`rejected` 表示截断或其他协议状态不完整。拒绝不等于成功，也不能通过换模型或改写资料绕过。
