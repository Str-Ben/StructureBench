#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thin wrappers around datasets.load_dataset for local files.
Keeps per-script data loading consistent and concise.
"""

from typing import Dict

from datasets import load_dataset


def load_parquet(split: str, data_file: str):
    ds_dict = load_dataset("parquet", data_files={split: data_file})
    assert split in ds_dict, f"split {split!r} 不在数据文件中，可用的有: {list(ds_dict.keys())}"
    return ds_dict[split]


def load_jsonl(split: str, data_file: str, field: str = None):
    """
    Load JSONL or JSON array (each line an object). Optional `field` if the file wraps data.
    """
    data_files: Dict[str, str] = {split: data_file}
    ds_dict = load_dataset("json", data_files=data_files, field=field)
    assert split in ds_dict, f"split {split!r} 不在数据文件中，可用的有: {list(ds_dict.keys())}"
    return ds_dict[split]


def load_json(split: str, data_file: str, field: str = None):
    """
    Alias for load_jsonl for clarity (datasets 'json' loader works for json/jsonl).
    """
    return load_jsonl(split=split, data_file=data_file, field=field)
