import argparse
from pathlib import Path
import sys

from .evaluation import evaluate
from .export import write_result
from .ingest import inventory_zip, extract_selected, parse_endnote, read_json, write_json


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AcuExactAgent | 先圈定文章，再逐字提取；出处与查证分列。")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run synthetic offline regression demo")
    demo.add_argument("--out", default="outputs/demo")
    server = commands.add_parser("serve", help="Start local browser demo")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    agent = commands.add_parser("agent", help="PDF + prompt -> multi-provider extraction, audit, classification")
    agent.add_argument("--pdf", required=True)
    task_prompt = agent.add_mutually_exclusive_group(required=True)
    task_prompt.add_argument("--prompt")
    task_prompt.add_argument("--prompt-file")
    agent.add_argument("--provider", choices=["openai", "anthropic", "minimax", "openai_compatible"], default="openai")
    agent.add_argument("--model")
    agent.add_argument("--base-url", default="")
    agent.add_argument("--api-key-env", help="Name of environment variable; never put a key in command history")
    agent.add_argument("--vision", action="store_true")
    agent.add_argument("--vision-provider", choices=["openai", "anthropic", "openai_compatible"])
    agent.add_argument("--vision-model")
    agent.add_argument("--vision-base-url", default="")
    agent.add_argument("--vision-key-env")
    agent.add_argument("--title", default="", help="Target article title from EndNote, especially for multi-article PDFs")
    agent.add_argument("--pages", default="", help="Physical PDF pages, e.g. 1-3,5")
    agent.add_argument("--max-pages", type=int, default=30)
    agent.add_argument("--max-calls", type=int, default=80)
    agent.add_argument("--timeout", type=float, default=60)
    agent.add_argument("--out", default="outputs/agent")
    inv = commands.add_parser("inventory")
    inv.add_argument("--zip", required=True); inv.add_argument("--out", required=True)
    select = commands.add_parser("unpack")
    select.add_argument("--zip", required=True); select.add_argument("--member", action="append", required=True)
    select.add_argument("--out", required=True)
    endnote = commands.add_parser("endnote")
    endnote.add_argument("--input", required=True); endnote.add_argument("--out", required=True)
    plan = commands.add_parser("plan", help="Suggest PDF matches for EndNote records; never auto-approve article bounds")
    plan.add_argument("--records", required=True); plan.add_argument("--inventory", required=True); plan.add_argument("--out", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--pdf", required=True); inspect.add_argument("--out", required=True)
    inspect.add_argument("--title", default=""); inspect.add_argument("--max-pages", type=int)
    extract = commands.add_parser("extract")
    extract.add_argument("--pdf", required=True); extract.add_argument("--manifest", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--ocr", action="store_true", help="Force OCR for every selected region; all OCR quotes require review")
    extract.add_argument("--llm", action="store_true"); extract.add_argument("--model")
    extract.add_argument("--catalog", help="Verified local source catalog JSON")
    extract.add_argument("--lookup", action="store_true", help="Search missing source on Chinese Wikisource; sends quote text, returns candidates for review")
    extract.add_argument("--disease-names", help="JSON list: exact short disease terms to exclude")
    batch = commands.add_parser("batch")
    batch.add_argument("--manifest", required=True); batch.add_argument("--out", required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--pred", required=True); ev.add_argument("--gold", required=True); ev.add_argument("--out", required=True)
    gold = commands.add_parser("gold-import")
    gold.add_argument("--input", required=True); gold.add_argument("--out", required=True)
    gold.add_argument("--article-column", default="文章ID"); gold.add_argument("--quote-column", default="古籍条文")
    gold.add_argument("--book-column"); gold.add_argument("--chapter-column"); gold.add_argument("--sheet")
    lookup = commands.add_parser("lookup", help="Look up missing source candidates online, without altering explicit citations")
    lookup.add_argument("--input", required=True); lookup.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .pipeline import demo as run_demo
            result = run_demo()
            write_result(result, args.out)
            gold_data = read_json(Path(__file__).parent / "data" / "demo-gold.json")
            write_json(Path(args.out) / "evaluation.json", {"dataset": "synthetic_fixture_only", "exact": evaluate(result["quotes"], gold_data)})
            print(f"Synthetic demo: {len(result['quotes'])} quotes -> {args.out}")
        elif args.command == "serve":
            from .server import serve
            serve(args.port, args.host)
        elif args.command == "agent":
            import os
            from .agent import run_pdf_agent
            from .pdf import open_pdf
            from .providers import ModelClient, ProviderConfig, provider_options
            from .workspace import parse_pages
            defaults = {p["id"]: p["default_model"] for p in provider_options()}
            def key(variable):
                if variable and not os.environ.get(variable):
                    raise ValueError(f"Environment variable {variable} is not set")
                return os.environ.get(variable, "") if variable else ""
            client = ModelClient(ProviderConfig(args.provider, args.model or defaults[args.provider], key(args.api_key_env), args.base_url, args.timeout, vision=args.vision))
            vision_client = ModelClient(ProviderConfig(args.vision_provider, args.vision_model or defaults[args.vision_provider], key(args.vision_key_env), args.vision_base_url, args.timeout, vision=True)) if args.vision_provider else None
            client._key()
            if vision_client:
                vision_client._key()
            with open_pdf(args.pdf) as pdf:
                pages = parse_pages(args.pages, len(pdf.pages))
            prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
            result = run_pdf_agent(args.pdf, prompt, client, target_title=args.title, page_numbers=pages,
                                   max_pages=args.max_pages, max_calls=args.max_calls, vision_client=vision_client,
                                   on_event=lambda event: print(f"[{event.get('stage', '')}] {event.get('message', '')}", flush=True))
            write_result(result, args.out)
            print(f"{len(result['quotes'])} quotes; status={result['status']} -> {args.out}")
            if result["status"] in {"needs_scope", "failed", "cancelled"}:
                return 1
        elif args.command == "inventory":
            report = inventory_zip(args.zip)
            write_json(args.out, report)
            print(f"{report['files']} files; {report['uncompressed_bytes']} uncompressed bytes")
        elif args.command == "unpack":
            paths = extract_selected(args.zip, args.member, args.out)
            print(f"Extracted {len(paths)} explicitly selected files")
        elif args.command == "endnote":
            records = parse_endnote(args.input)
            write_json(args.out, records)
            print(f"{len(records)} EndNote records")
        elif args.command == "plan":
            from .planner import plan_records
            plans = plan_records(read_json(args.records), read_json(args.inventory))
            write_json(args.out, plans)
            print(f"{len(plans)} article plans; all boundaries pending review")
        elif args.command == "inspect":
            from .pdf import inspect_pdf
            report = inspect_pdf(args.pdf, args.out, args.title, args.max_pages)
            print(f"{len(report['pages'])} page previews; boundaries require review")
        elif args.command == "extract":
            from .pipeline import analyze_pdf
            result = analyze_pdf(args.pdf, read_json(args.manifest), ocr=args.ocr,
                                 use_llm=args.llm, model=args.model,
                                 lookup=args.lookup,
                                 catalog=read_json(args.catalog) if args.catalog else None,
                                 disease_names=read_json(args.disease_names) if args.disease_names else None)
            write_result(result, args.out)
            print(f"{len(result['quotes'])} quotes; status={result['status']} -> {args.out}")
        elif args.command == "batch":
            from .pipeline import analyze_pdf
            source = Path(args.manifest).resolve()
            jobs = read_json(source)
            if not isinstance(jobs, list):
                raise ValueError("Batch manifest must be a list of {pdf, manifest} jobs")
            reports, quotes = [], []
            article_ids = set()
            for index, job in enumerate(jobs):
                try:
                    manifest = read_json(source.parent / job["manifest"])
                    aid = str(manifest["article"]["id"])
                    if aid in article_ids:
                        raise ValueError("Duplicate article ID in batch")
                    result = analyze_pdf(source.parent / job["pdf"], manifest)
                    article_ids.add(aid)
                    write_result(result, Path(args.out) / f"article-{index + 1:04d}")
                    quotes.extend(result["quotes"])
                    reports.append({"job": index + 1, "article_id": aid, "status": result["status"], "quotes": len(result["quotes"])})
                except Exception as exc:
                    reports.append({"job": index + 1, "status": "failed", "error": str(exc)})
            write_result({"quotes": quotes, "jobs": reports}, args.out)
            print(f"{len(reports)} jobs; {sum(r['status'] == 'failed' for r in reports)} failed")
            if any(r["status"] == "failed" for r in reports):
                return 1
        elif args.command == "evaluate":
            predictions, annotations = read_json(args.pred), read_json(args.gold)
            predictions = predictions["quotes"] if isinstance(predictions, dict) else predictions
            annotations = annotations["quotes"] if isinstance(annotations, dict) else annotations
            write_json(args.out, {"exact": evaluate(predictions, annotations),
                                  "whitespace_insensitive": evaluate(predictions, annotations, layout_only=True)})
            print(f"Evaluation -> {args.out}")
        elif args.command == "gold-import":
            from .gold import import_gold
            result = import_gold(args.input, args.article_column, args.quote_column, args.book_column, args.chapter_column, args.sheet)
            write_json(args.out, result)
            print(f"{len(result)} gold occurrences imported; content unchanged")
        elif args.command == "lookup":
            from .retrieval import lookup_source
            result = read_json(args.input)
            if not isinstance(result, dict) or not isinstance(result.get("quotes"), list):
                raise ValueError("lookup expects a result.json object with quotes")
            result["quotes"] = [lookup_source(q, enable_network=True) if not q.get("source_explicit", {}).get("book") or not q.get("source_explicit", {}).get("chapter") else q for q in result["quotes"]]
            write_result(result, args.out)
            print(f"Source candidates saved to {args.out}; review required")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
