"""Resumable PDF-agent batches with process isolation and immediate disk output."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time
import zipfile

from .export import to_tsv, write_result
from .providers import ModelClient, ProviderConfig, _check_schema


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def natural_key(value):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)]


class CheckpointClient(ModelClient):
    """Keep validated successful stages so interrupted papers do not start over."""
    def __init__(self, config, directory):
        super().__init__(config)
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

    def generate_json(self, system, payload, schema, *, images=None, schema_name="agent_result"):
        signature = {"system": system, "payload": payload, "schema": schema, "images": images,
                     "image_detail": "high",
                     "provider": self.config.provider, "model": self.config.model,
                     "base_url": self.base_url, "max_tokens": self.config.max_tokens,
                     "reasoning_effort": self.config.reasoning_effort}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.directory / (digest + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            _check_schema(schema).validate(result)
            self.calls.append({"provider": self.config.provider, "model": self.config.model,
                               "status": "cache_hit", "usage": {}, "request_id": None})
            return result
        result = super().generate_json(system, payload, schema, images=images, schema_name=schema_name)
        atomic_json(path, result)
        return result


def pipeline_complete(result):
    incomplete = {"agent_incomplete", "call_budget_exceeded", "cancelled", "quote_review_failed",
                  "quote_review_incomplete", "vision_required", "unreadable_region_requires_vision"}
    return (result.get("status") in {"completed", "needs_review"}
            and "review_quotes" in result.get("audit", {}).get("stages", [])
            and not any(w.get("code") in incomplete for w in result.get("warnings", []))
            and all(q.get("relevance", {}).get("method") == "llm_agent"
                    for q in result.get("quotes", [])))


def select_jobs(source, limit, start=1, prefix=""):
    source = Path(source).resolve()
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [i for i in archive.infolist() if not i.is_dir()
                       and i.filename.lower().endswith(".pdf")
                       and not i.filename.startswith("__MACOSX/")
                       and i.filename.startswith(prefix)]
            members.sort(key=lambda i: natural_key(i.filename))
            selected = members[start - 1:start - 1 + limit]
            return [{"member": i.filename, "size": i.file_size, "crc": i.CRC} for i in selected]
    if source.is_dir():
        files = sorted((p for p in source.rglob("*") if p.is_file()
                        and p.suffix.lower() == ".pdf" and str(p.relative_to(source)).startswith(prefix)),
                       key=lambda p: natural_key(str(p.relative_to(source))))
        return [{"member": str(p.relative_to(source)), "size": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns} for p in files[start - 1:start - 1 + limit]]
    raise ValueError("Batch source must be a ZIP archive or a PDF directory")


def _run_one(job, settings):
    from .agent import run_pdf_agent
    directory = Path(settings["out"]) / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def event(value):
        atomic_json(directory / "status.json", dict(job, state="running", event=value,
                                                     started_at=started, updated_at=time.time()))
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=time.time(), **value), ensure_ascii=False) + "\n")

    event({"stage": "start", "message": "开始处理", "progress": 0})
    try:
        previous = directory / "result.json"
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            # JSON/schema-valid outputs can still fail semantic checks. Do not
            # replay a rejected plan or classification indefinitely on resume.
            if not pipeline_complete(old):
                for checkpoint in (directory / "checkpoints").glob("*.json"):
                    value = json.loads(checkpoint.read_text(encoding="utf-8"))
                    invalid_plan = "regions" in value and not old.get("audit", {}).get("regions")
                    if invalid_plan or "results" in value or "missing_quotes" in value:
                        checkpoint.unlink()
        client = CheckpointClient(ProviderConfig(**settings["config"]), directory / "checkpoints")
        result = run_pdf_agent(job["pdf"], settings["prompt"], client,
                               target_title=job["title"], max_pages=settings["max_pages"],
                               max_calls=settings["max_calls"], on_event=event,
                               transcribe_pages=False,
                               cancelled=lambda: (Path(settings["out"]) / "STOP").exists())
        # Write complete payloads first; status.json is the commit marker for resume.
        write_result(result, directory)
        complete = pipeline_complete(result)
        calls = result.get("audit", {}).get("model_calls", [])
        record = dict(job, state="done" if complete else "needs_attention",
                      pipeline_complete=complete, result_status=result["status"],
                      quotes=len(result["quotes"]), calls=result.get("audit", {}).get("calls_used", len(calls)),
                      cached_calls=sum(c.get("status") == "cache_hit" for c in calls),
                      transport_failure=bool(calls) and all(c.get("status") in {
                          "http_error", "network_error", "invalid_response"} for c in calls),
                      usage={key: sum(c.get("usage", {}).get(key, 0) for c in calls)
                             for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                      warnings=result.get("warnings", []),
                      seconds=round(time.time() - started, 2), updated_at=time.time())
    except Exception as exc:
        # Arbitrary exception text can contain credentials or transport bodies.
        record = dict(job, state="failed", pipeline_complete=False,
                      error_type=type(exc).__name__, seconds=round(time.time() - started, 2),
                      updated_at=time.time())
    atomic_json(directory / "status.json", record)
    return record


def run_batch(source, out, prompt, config, *, limit=350, start=1, prefix="", workers=6,
              max_pages=20, max_calls=80):
    if not 1 <= workers <= 32 or limit < 1 or start < 1:
        raise ValueError("workers must be 1–32; limit/start must be positive")
    if "api_key" in config and config["api_key"]:
        raise ValueError("Batch credentials must come from the provider environment variable")
    ModelClient(ProviderConfig(**config))._key()
    out, source = Path(out).resolve(), Path(source).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = select_jobs(source, limit, start, prefix)
    if not selected:
        raise ValueError("No PDFs match this selection")
    fingerprint = {"source": str(source), "selection": selected, "prompt": prompt, "config": config,
                   "max_pages": max_pages, "max_calls": max_calls}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Existing batch differs from this source/config/prompt; use another output directory")
        jobs = old["jobs"]
    else:
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        jobs = []
        archive = zipfile.ZipFile(source) if source.is_file() else None
        try:
            for index, item in enumerate(selected, start):
                identifier = f"article-{index:04d}-{hashlib.sha256(item['member'].encode()).hexdigest()[:8]}"
                destination = inputs / (identifier + ".pdf")
                if archive:
                    # Never use extract/extractall: ZIP member names cannot escape inputs/.
                    destination.write_bytes(archive.read(item["member"]))
                else:
                    destination = source / item["member"]
                title = re.sub(r"^\d+[\s._-]*", "", Path(item["member"]).stem).strip()
                title = re.sub(r"\s+\d+$", "", title)  # Download duplicate suffix, e.g. "paper 2.pdf".
                jobs.append(dict(id=identifier, index=index, member=item["member"],
                                 pdf=str(destination), title=title))
        finally:
            if archive:
                archive.close()
        atomic_json(manifest_path, {"fingerprint": fingerprint, "jobs": jobs})
    (out / "research-prompt.txt").write_text(prompt, encoding="utf-8")
    settings = dict(out=str(out), prompt=prompt, config=config, max_pages=max_pages, max_calls=max_calls)
    records = {}
    for job in jobs:
        status = out / job["id"] / "status.json"
        if status.exists() and (out / job["id"] / "result.json").exists():
            record = json.loads(status.read_text(encoding="utf-8"))
            if record.get("pipeline_complete"):
                records[job["id"]] = record
    pending = iter(job for job in jobs if job["id"] not in records)

    def publish(state):
        quotes = []
        for job in jobs:
            if job["id"] in records:
                result_path = out / job["id"] / "result.json"
                if result_path.exists():
                    quotes.extend(json.loads(result_path.read_text(encoding="utf-8"))["quotes"])
        temporary = out / "quotes.tsv.tmp"
        temporary.write_text(to_tsv(quotes), encoding="utf-8-sig")
        os.replace(temporary, out / "quotes.tsv")
        atomic_json(out / "results.json", {"quotes": quotes, "jobs": list(records.values())})
        atomic_json(out / "progress.json", {
            "state": state, "total": len(jobs), "finished": len(records),
            "pipeline_completed": sum(bool(r.get("pipeline_complete")) for r in records.values()),
            "needs_attention": sum(not r.get("pipeline_complete", False) for r in records.values()),
            "quotes": len(quotes), "updated_at": time.time(), "workers": workers,
            "jobs": [records.get(j["id"], dict(j, state="pending_or_running")) for j in jobs]})

    publish("running")
    consecutive_failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}

        def submit():
            job = next(pending, None)
            if job is not None:
                futures[pool.submit(_run_one, job, settings)] = job

        for _ in range(workers):
            submit()
        while futures:
            done, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    record = dict(job, state="failed", pipeline_complete=False, error_type=type(exc).__name__)
                records[job["id"]] = record
                consecutive_failures = consecutive_failures + 1 if record["state"] == "failed" or record.get("transport_failure") else 0
                print(f"[{len(records)}/{len(jobs)}] {record['state']} | {job['member']} | quotes={record.get('quotes', 0)}", flush=True)
                publish("running")
                if not (out / "STOP").exists() and consecutive_failures < max(3, workers):
                    submit()
    stopped = len(records) < len(jobs)
    publish("stopped" if stopped else "finished")
    return 1 if stopped or any(not r.get("pipeline_complete") for r in records.values()) else 0


def configure_parser(commands):
    parser = commands.add_parser("agent-batch", help="Run PDF agents in parallel; save each result immediately; resume completed jobs")
    parser.add_argument("--source", required=True, help="ZIP archive or PDF directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--start", type=int, default=1, help="One-based natural-sort position")
    parser.add_argument("--prefix", default="", help="Restrict ZIP members/relative PDF paths")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--provider", default="minimax", choices=["minimax", "openai", "anthropic", "openai_compatible"])
    parser.add_argument("--model", default="MiniMax-M3.1-Flash")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--text-only", action="store_true")


def from_args(args):
    return run_batch(args.source, args.out, Path(args.prompt_file).read_text(encoding="utf-8-sig"),
                     dict(provider=args.provider, model=args.model, base_url=args.base_url,
                          vision=not args.text_only, reasoning_effort=args.reasoning_effort,
                          timeout=args.timeout, max_tokens=args.max_tokens),
                     limit=args.limit, start=args.start, prefix=args.prefix, workers=args.workers,
                     max_pages=args.max_pages, max_calls=args.max_calls)
