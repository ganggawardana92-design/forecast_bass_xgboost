"""
Preprocessing data mentah ANALIS.xls untuk tesis Hybrid Bass + XGBoost.

PENTING:
- File ANALIS.xls sebenarnya berformat dBase/DBF, bukan workbook Excel.
- TIPE='J' dipakai sebagai transaksi penjualan/konsumsi.
- QTY negatif pada TIPE J dipertahankan sebagai net sales dan juga dicatat sebagai return/correction candidate.
- B/D/K tidak dimasukkan ke target penjualan.
- Outlier tidak dihapus otomatis; hanya ditandai untuk audit.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])

ensure_package("pandas", "pandas")
ensure_package("dbfread", "dbfread")

import numpy as np
import pandas as pd
from dbfread import DBF


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="ANALIS.xls", help="Path file ANALIS.xls/DBF")
    p.add_argument("--output", default="data/daily_sales_jual.csv", help="CSV harian hasil preprocessing")
    p.add_argument("--summary", default="data/preprocessing_summary.csv", help="Ringkasan preprocessing")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)
    summary_path = Path(args.summary)

    if not src.exists():
        raise FileNotFoundError(
            f"{src} tidak ditemukan. Upload ANALIS.xls ke Colab, lalu jalankan kembali."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    type_counts = Counter()
    invalid_date = 0
    invalid_qty = 0
    j_negative_rows = 0
    j_zero_rows = 0

    daily_qty = defaultdict(float)
    daily_positive_qty = defaultdict(float)
    daily_return_abs = defaultdict(float)
    daily_rows = Counter()

    print("Membaca DBF secara streaming...")
    table = DBF(
        str(src),
        load=False,
        char_decode_errors="ignore",
        ignore_missing_memofile=True,
    )

    for rec in table:
        tipe = str(rec.get("TIPE", "")).strip().upper()
        type_counts[tipe] += 1

        if tipe != "J":
            continue

        date = pd.to_datetime(rec.get("TGL"), errors="coerce")
        if pd.isna(date):
            invalid_date += 1
            continue

        qty = pd.to_numeric(rec.get("QTY"), errors="coerce")
        if pd.isna(qty):
            invalid_qty += 1
            continue

        date = pd.Timestamp(date).normalize()
        qty = float(qty)

        daily_qty[date] += qty
        daily_rows[date] += 1

        if qty > 0:
            daily_positive_qty[date] += qty
        elif qty < 0:
            j_negative_rows += 1
            daily_return_abs[date] += abs(qty)
        else:
            j_zero_rows += 1

    if not daily_qty:
        raise ValueError("Tidak ditemukan transaksi TIPE='J' yang valid.")

    min_date = min(daily_qty)
    max_date = max(daily_qty)
    calendar = pd.date_range(min_date, max_date, freq="D")

    daily = pd.DataFrame({"date": calendar})
    daily["sales_qty"] = daily["date"].map(daily_qty).fillna(0.0).astype(float)
    daily["gross_positive_qty"] = daily["date"].map(daily_positive_qty).fillna(0.0).astype(float)
    daily["return_or_correction_qty_abs"] = daily["date"].map(daily_return_abs).fillna(0.0).astype(float)
    daily["transaction_rows"] = daily["date"].map(daily_rows).fillna(0).astype(int)
    daily["is_zero_day"] = (daily["sales_qty"] == 0).astype(int)

    # Tandai outlier, jangan dihapus otomatis.
    q1 = daily["sales_qty"].quantile(0.25)
    q3 = daily["sales_qty"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    daily["is_iqr_outlier"] = (
        (daily["sales_qty"] < lower) | (daily["sales_qty"] > upper)
    ).astype(int)

    daily.to_csv(out, index=False)

    summary_rows = [
        ("source_file", str(src)),
        ("source_format", "dBase/DBF (meskipun ekstensi .xls)"),
        ("date_min", str(min_date.date())),
        ("date_max", str(max_date.date())),
        ("calendar_days", len(daily)),
        ("TIPE_J_rows", type_counts.get("J", 0)),
        ("TIPE_B_rows", type_counts.get("B", 0)),
        ("TIPE_D_rows", type_counts.get("D", 0)),
        ("TIPE_K_rows", type_counts.get("K", 0)),
        ("J_negative_qty_rows", j_negative_rows),
        ("J_zero_qty_rows", j_zero_rows),
        ("invalid_J_date_rows", invalid_date),
        ("invalid_J_qty_rows", invalid_qty),
        ("zero_sales_days", int(daily["is_zero_day"].sum())),
        ("iqr_outlier_days", int(daily["is_iqr_outlier"].sum())),
        ("total_net_sales_qty", float(daily["sales_qty"].sum())),
        ("total_positive_sales_qty", float(daily["gross_positive_qty"].sum())),
        ("total_negative_J_abs_qty", float(daily["return_or_correction_qty_abs"].sum())),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(summary_path, index=False)

    print("\nPREPROCESSING SELESAI")
    print(f"Output data  : {out}")
    print(f"Output audit : {summary_path}")
    print(f"Periode      : {min_date.date()} s.d. {max_date.date()}")
    print(f"TIPE counts  : {dict(type_counts)}")
    print(f"J negatif    : {j_negative_rows} baris")
    print(f"Zero days    : {int(daily['is_zero_day'].sum())}")
    print(f"Outlier IQR  : {int(daily['is_iqr_outlier'].sum())}")
    print("\nCatatan:")
    print("- B/D/K tidak masuk ke target penjualan.")
    print("- QTY negatif pada J tidak dibuang; dicek sebagai kandidat retur/koreksi.")
    print("- Hari nol dan outlier tidak dihapus otomatis karena bisa merupakan kejadian nyata.")


if __name__ == "__main__":
    main()
