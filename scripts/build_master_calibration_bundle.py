#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-lane REAP calibration bundle from local JSONL data plus "
            "optional public Hugging Face datasets."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the bundle config JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL and summary outputs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned bundle summary without downloading public datasets or materializing JSONL rows.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class BuiltRow:
    row_id: str
    text: str
    estimated_tokens: int
    lane: str
    source_kind: str
    source_dataset: str
    source_config: str | None
    source_label: str
    extra: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "text": self.text,
            "estimated_tokens": self.estimated_tokens,
            "lane": self.lane,
            "source_kind": self.source_kind,
            "source_dataset": self.source_dataset,
            "source_config": self.source_config,
            "source_label": self.source_label,
            **self.extra,
        }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text).strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def render_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip()
        content = render_message_content(message.get("content"))
        if not content:
            continue
        rendered.append(f"{role}: {content}")
    return "\n\n".join(rendered)


def extract_local_text(payload: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages_text = render_messages(payload.get("messages"))
    if messages_text:
        return messages_text
    # Fallback: join string-like fields so unknown chat exports still render.
    fallback_parts: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str) and value.strip():
            fallback_parts.append(f"{key}: {value.strip()}")
    return "\n\n".join(fallback_parts)


def iter_local_rows(local_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(local_cfg["path"])
    text_fields = local_cfg.get("text_fields") or ["prompt_text", "text", "content"]
    id_field = local_cfg.get("id_field", "prompt_id")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
            text = extract_local_text(payload, text_fields).strip()
            if not text:
                continue
            row_id = str(payload.get(id_field) or f"local-{line_no}")
            rows.append(
                {
                    "row_id": row_id,
                    "text": text,
                    "estimated_tokens": estimate_tokens(text),
                    "payload": payload,
                }
            )
    return rows


def select_local_rows(
    rows: list[dict[str, Any]],
    *,
    selection: str,
    max_rows: int | None,
    max_total_tokens: int | None,
    max_tokens_per_row: int | None,
    seed: int,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_tokens_per_row = kwargs.get("min_tokens_per_row")
    filtered = list(rows)
    if max_tokens_per_row is not None:
        filtered = [row for row in filtered if row["estimated_tokens"] <= max_tokens_per_row]
    if min_tokens_per_row is not None:
        filtered = [row for row in filtered if row["estimated_tokens"] >= min_tokens_per_row]

    if selection == "longest":
        filtered.sort(key=lambda row: (-row["estimated_tokens"], row["row_id"]))
    elif selection == "shortest":
        filtered.sort(key=lambda row: (row["estimated_tokens"], row["row_id"]))
    elif selection == "random":
        rng = random.Random(seed)
        rng.shuffle(filtered)
    else:
        filtered.sort(key=lambda row: row["row_id"])

    chosen: list[dict[str, Any]] = []
    total_tokens = 0
    for row in filtered:
        if max_rows is not None and len(chosen) >= max_rows:
            break
        if max_total_tokens is not None and total_tokens + row["estimated_tokens"] > max_total_tokens:
            continue
        chosen.append(row)
        total_tokens += row["estimated_tokens"]

    summary = {
        "selection": selection,
        "available_rows": len(filtered),
        "selected_rows": len(chosen),
        "selected_tokens": total_tokens,
    }
    return chosen, summary


def row_to_text(dataset_name: str, dataset_config: str | None, row: dict[str, Any], text_fields: list[str] | None) -> str:
    if dataset_name == "theblackcat102/evol-codealpaca-v1":
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("output") or "").strip()
        return f"Instruction:\n{instruction}\n\nResponse:\n{output}".strip()

    if dataset_name == "Salesforce/xlam-function-calling-60k":
        query = str(row.get("query") or "").strip()
        tools = json.dumps(row.get("tools") or [], ensure_ascii=False)
        answers = json.dumps(row.get("answers") or [], ensure_ascii=False)
        return f"Query:\n{query}\n\nTools:\n{tools}\n\nAnswers:\n{answers}".strip()

    if dataset_name == "SWE-bench/SWE-smith-trajectories":
        fields = ["problem_statement", "instance_id", "text", "trajectory"]
        parts = [f"{field}: {row.get(field)}" for field in fields if row.get(field)]
        return "\n\n".join(parts).strip()

    if dataset_name == "open-r1/Mixture-of-Thoughts":
        rendered = render_messages(row.get("messages"))
        if rendered:
            return rendered

    if text_fields:
        parts = []
        for field in text_fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(f"{field}: {value.strip()}")
        if parts:
            return "\n\n".join(parts)

    return json.dumps(row, ensure_ascii=False)


def load_hf_dataset_rows(source_cfg: dict[str, Any], lane_name: str, seed: int) -> tuple[list[BuiltRow], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required. Use `uv run --with datasets ...`.") from exc

    dataset_name = source_cfg["dataset"]
    dataset_config = source_cfg.get("config")
    split = source_cfg.get("split", "train")
    rows_requested = int(source_cfg["rows"])
    selection = source_cfg.get("selection", "random")
    text_fields = source_cfg.get("text_fields")
    label = source_cfg.get("label", dataset_name)
    max_tokens_per_row = source_cfg.get("max_tokens_per_row")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    min_tokens_per_row = source_cfg.get("min_tokens_per_row")

    indices = list(range(len(dataset)))
    if selection == "random":
        rng = random.Random(seed)
        rng.shuffle(indices)
    elif selection == "first":
        pass
    elif selection == "longest":
        # Pre-scan to sort by text length descending
        lengths = []
        for i in indices:
            text = row_to_text(dataset_name, dataset_config, dataset[i], text_fields).strip()
            lengths.append((i, len(text)))
        lengths.sort(key=lambda x: -x[1])
        indices = [i for i, _ in lengths]
    else:
        raise ValueError(f"Unsupported public selection {selection!r}")

    built_rows: list[BuiltRow] = []
    picked = 0
    total_tokens = 0
    for index in indices:
        if picked >= rows_requested:
            break
        payload = dataset[int(index)]
        text = row_to_text(dataset_name, dataset_config, payload, text_fields).strip()
        if not text:
            continue
        estimated_tokens = estimate_tokens(text)
        if max_tokens_per_row is not None and estimated_tokens > max_tokens_per_row:
            continue
        if min_tokens_per_row is not None and estimated_tokens < min_tokens_per_row:
            continue
        built_rows.append(
            BuiltRow(
                row_id=f"{label}-{index}",
                text=text,
                estimated_tokens=estimated_tokens,
                lane=lane_name,
                source_kind="huggingface",
                source_dataset=dataset_name,
                source_config=dataset_config,
                source_label=label,
                extra={"source_index": index},
            )
        )
        total_tokens += estimated_tokens
        picked += 1

    summary = {
        "label": label,
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "selection": selection,
        "rows_requested": rows_requested,
        "rows_built": len(built_rows),
        "estimated_tokens": total_tokens,
    }
    return built_rows, summary


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def write_bundle_markdown(path: Path, summary: dict[str, Any]) -> None:
    lane_rows: list[list[Any]] = []
    for lane in summary["lanes"]:
        lane_rows.append(
            [
                lane["lane"],
                lane["status"],
                lane["planned_rows"],
                lane["built_rows"],
                lane["estimated_tokens"],
                lane["output_jsonl"] or "n/a",
            ]
        )

    source_rows: list[list[Any]] = []
    for lane in summary["lanes"]:
        for source in lane["sources"]:
            source_rows.append(
                [
                    lane["lane"],
                    source["source"],
                    source["selection"],
                    source["rows_requested"],
                    source["rows_built"],
                    source["estimated_tokens"],
                ]
            )

    public_mix_rows: list[list[Any]] = []
    for lane in summary["lanes"]:
        for source in lane["sources"]:
            if source["source_kind"] != "huggingface":
                continue
            public_mix_rows.append(
                [
                    source["source"],
                    source.get("config") or "default",
                    source["rows_requested"],
                    source["rows_built"],
                ]
            )

    sections = [
        f"# {summary['bundle_name']}",
        "",
        f"- Status: `{summary['status']}`",
        f"- Local dataset: `{summary['local_dataset_path']}`",
        f"- Output directory: `{summary['output_dir']}`",
        f"- Total planned rows: `{summary['totals']['planned_rows']}`",
        f"- Total built rows: `{summary['totals']['built_rows']}`",
        f"- Total estimated tokens: `{summary['totals']['estimated_tokens']}`",
        "",
        "## Lane summary",
        markdown_table(
            ["Lane", "Status", "Planned rows", "Built rows", "Estimated tokens", "Output JSONL"],
            lane_rows,
        ),
        "",
        "## Source breakdown",
        markdown_table(
            ["Lane", "Source", "Selection", "Rows requested", "Rows built", "Estimated tokens"],
            source_rows,
        ),
    ]
    if public_mix_rows:
        sections.extend(
            [
                "",
                "## Public mix",
                markdown_table(["Dataset", "Config", "Rows requested", "Rows built"], public_mix_rows),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def build_bundle(config: dict[str, Any], output_dir: Path, dry_run: bool, config_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 7))
    bundle_name = str(config["name"])
    local_cfg = dict(config["local_dataset"])
    local_dataset_path = str(local_cfg["path"])
    local_path = Path(local_dataset_path)
    if not local_path.is_absolute():
        local_path = (config_dir / local_path).resolve()
    local_cfg["path"] = str(local_path)
    local_rows = [] if dry_run else iter_local_rows(local_cfg)

    bundle_summary: dict[str, Any] = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "status": "dry_run" if dry_run else "ok",
        "seed": seed,
        "local_dataset_path": local_dataset_path,
        "output_dir": str(output_dir),
        "lanes": [],
        "totals": {"planned_rows": 0, "built_rows": 0, "estimated_tokens": 0},
    }
    merged_rows: list[dict[str, Any]] = []

    for lane_index, lane_cfg in enumerate(config["lanes"], start=1):
        lane_name = lane_cfg["name"]
        lane_output = output_dir / f"{lane_name}.jsonl"
        lane_summary = {
            "lane": lane_name,
            "status": "dry_run" if dry_run else "ok",
            "description": lane_cfg.get("description", ""),
            "output_jsonl": None if dry_run else str(lane_output),
            "sources": [],
            "planned_rows": 0,
            "built_rows": 0,
            "estimated_tokens": 0,
        }
        lane_rows: list[dict[str, Any]] = []

        for source_index, source_cfg in enumerate(lane_cfg["sources"], start=1):
            source_kind = source_cfg["type"]
            source_seed = seed + lane_index * 100 + source_index
            if source_kind == "local":
                rows_requested = int(source_cfg.get("max_rows") or 0)
                if dry_run:
                    estimated_tokens = int(source_cfg.get("max_total_tokens") or 0)
                    source_summary = {
                        "source_kind": "local_jsonl",
                        "source": source_cfg.get("label", "local"),
                        "selection": source_cfg.get("selection", "first"),
                        "rows_requested": rows_requested,
                        "rows_built": rows_requested,
                        "estimated_tokens": estimated_tokens,
                        "input_rows_hint": int(local_cfg.get("input_rows_hint") or 0),
                    }
                else:
                    selected, local_summary = select_local_rows(
                        local_rows,
                        selection=source_cfg.get("selection", "first"),
                        max_rows=source_cfg.get("max_rows"),
                        max_total_tokens=source_cfg.get("max_total_tokens"),
                        max_tokens_per_row=source_cfg.get("max_tokens_per_row"),
                        min_tokens_per_row=source_cfg.get("min_tokens_per_row"),
                        seed=source_seed,
                    )
                    source_summary = {
                        "source_kind": "local_jsonl",
                        "source": source_cfg.get("label", "local"),
                        "selection": local_summary["selection"],
                        "rows_requested": rows_requested,
                        "rows_built": local_summary["selected_rows"],
                        "estimated_tokens": local_summary["selected_tokens"],
                    }
                    for row in selected:
                        built = BuiltRow(
                            row_id=row["row_id"],
                            text=row["text"],
                            estimated_tokens=row["estimated_tokens"],
                            lane=lane_name,
                            source_kind="local_jsonl",
                            source_dataset="local_jsonl",
                            source_config=None,
                            source_label=source_cfg.get("label", "local"),
                            extra={},
                        )
                        lane_rows.append(built.to_json())
            elif source_kind == "huggingface":
                source_summary = {
                    "source_kind": "huggingface",
                    "source": source_cfg.get("label", source_cfg["dataset"]),
                    "config": source_cfg.get("config"),
                    "selection": source_cfg.get("selection", "random"),
                    "rows_requested": int(source_cfg["rows"]),
                    "rows_built": int(source_cfg["rows"]) if dry_run else None,
                    "estimated_tokens": 0,
                }
                if not dry_run:
                    built_rows, hf_summary = load_hf_dataset_rows(source_cfg, lane_name, source_seed)
                    source_summary.update(
                        {
                            "rows_built": hf_summary["rows_built"],
                            "estimated_tokens": hf_summary["estimated_tokens"],
                        }
                    )
                    lane_rows.extend([row.to_json() for row in built_rows])
            elif source_kind == "external":
                ext_path = Path(source_cfg["path"])
                if not ext_path.is_absolute():
                    ext_path = (config_dir / ext_path).resolve()
                rows_requested = int(source_cfg.get("rows") or 0)
                label = source_cfg.get("label", ext_path.stem)
                text_field = source_cfg.get("text_field", "text")
                if dry_run:
                    source_summary = {
                        "source_kind": "external_jsonl",
                        "source": label,
                        "selection": "all",
                        "rows_requested": rows_requested,
                        "rows_built": rows_requested,
                        "estimated_tokens": 0,
                    }
                else:
                    ext_rows: list[dict[str, Any]] = []
                    with ext_path.open("r", encoding="utf-8") as handle:
                        for line_no, line in enumerate(handle, start=1):
                            raw = line.strip()
                            if not raw:
                                continue
                            payload = json.loads(raw)
                            text = str(payload.get(text_field, "")).strip()
                            if not text:
                                continue
                            et = estimate_tokens(text)
                            built = BuiltRow(
                                row_id=f"{label}-{line_no}",
                                text=text,
                                estimated_tokens=et,
                                lane=lane_name,
                                source_kind="external_jsonl",
                                source_dataset=str(ext_path),
                                source_config=None,
                                source_label=label,
                                extra={},
                            )
                            ext_rows.append(built.to_json())
                            if rows_requested and len(ext_rows) >= rows_requested:
                                break
                    total_ext_tokens = sum(r["estimated_tokens"] for r in ext_rows)
                    source_summary = {
                        "source_kind": "external_jsonl",
                        "source": label,
                        "selection": "all",
                        "rows_requested": rows_requested or len(ext_rows),
                        "rows_built": len(ext_rows),
                        "estimated_tokens": total_ext_tokens,
                    }
                    lane_rows.extend(ext_rows)
            else:
                raise ValueError(f"Unsupported source type {source_kind!r}")

            lane_summary["sources"].append(source_summary)
            lane_summary["planned_rows"] += source_summary["rows_requested"]
            lane_summary["built_rows"] += int(source_summary["rows_built"] or 0)
            lane_summary["estimated_tokens"] += int(source_summary["estimated_tokens"] or 0)

        if not dry_run:
            write_jsonl(lane_output, lane_rows)
            merged_rows.extend(lane_rows)

        bundle_summary["lanes"].append(lane_summary)
        bundle_summary["totals"]["planned_rows"] += lane_summary["planned_rows"]
        bundle_summary["totals"]["built_rows"] += lane_summary["built_rows"]
        bundle_summary["totals"]["estimated_tokens"] += lane_summary["estimated_tokens"]

    if not dry_run:
        merged_path = output_dir / f"{bundle_name}.all.jsonl"
        write_jsonl(merged_path, merged_rows)
        bundle_summary["merged_output_jsonl"] = str(merged_path)
    else:
        bundle_summary["merged_output_jsonl"] = None

    return bundle_summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_bundle(config, output_dir, args.dry_run, config_path.parent)
    summary_json = output_dir / f"{summary['bundle_name']}.summary.json"
    summary_md = output_dir / f"{summary['bundle_name']}.summary.md"
    write_json(summary_json, summary)
    write_bundle_markdown(summary_md, summary)
    print(summary_json)
    print(summary_md)
    if summary.get("merged_output_jsonl"):
        print(summary["merged_output_jsonl"])


if __name__ == "__main__":
    main()
