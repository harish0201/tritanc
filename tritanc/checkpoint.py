"""Checkpoint save/load helpers for parquet, JSON, and TNF matrices."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ckpt_path(ckpt_dir: str, name: str, ext: str) -> str:
    return os.path.join(ckpt_dir, f"{name}.{ext}")


def _ckpt_exists(ckpt_dir: str, name: str, ext: str) -> bool:
    return os.path.exists(_ckpt_path(ckpt_dir, name, ext))


def save_parquet(df: pd.DataFrame, ckpt_dir: str, name: str) -> None:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(json.dumps)
    df.to_parquet(_ckpt_path(ckpt_dir, name, "parquet"))


def load_parquet(ckpt_dir: str, name: str, listcols=None) -> pd.DataFrame:
    df = pd.read_parquet(_ckpt_path(ckpt_dir, name, "parquet"))
    for col in listcols or []:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def save_json(obj: object, ckpt_dir: str, name: str) -> None:
    def default(o):
        if isinstance(o, set):
            return list(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")
    with open(_ckpt_path(ckpt_dir, name, "json"), "w") as fh:
        json.dump(obj, fh, default=default)


def load_json(ckpt_dir: str, name: str) -> object:
    with open(_ckpt_path(ckpt_dir, name, "json")) as fh:
        return json.load(fh)


def save_tnf(tnf: dict, ckpt_dir: str) -> None:
    contigs = list(tnf.keys())
    matrix = np.stack([tnf[c] for c in contigs]).astype(np.float32)
    df = pd.DataFrame(matrix, index=pd.Index(contigs, name="contig"))
    df.to_parquet(_ckpt_path(ckpt_dir, "tnf", "parquet"))


def load_tnf(ckpt_dir: str) -> dict:
    df = pd.read_parquet(_ckpt_path(ckpt_dir, "tnf", "parquet"))
    return {contig: row.values.astype(np.float32) for contig, row in df.iterrows()}


__all__ = [
    "_ckpt_path", "_ckpt_exists",
    "save_parquet", "load_parquet",
    "save_json", "load_json",
    "save_tnf", "load_tnf",
]
