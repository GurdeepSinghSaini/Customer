Customer Behavior Analytics
- Load Excel/CSV transactions
- Compute KPIs, RFM, cohort retention
- Export an Excel report with charts and CSVs

Required columns (case-insensitive; can be mapped via CLI):
- customer_id
- order_id
- order_date
- amount

Optional columns:
- product

Usage examples:
python customer_behavior_analytics.py --input data/transactions/*.xlsx --out outputs/customer_behavior_report.xlsx
python customer_behavior_analytics.py --input data/merged_sales.xlsx --sheet Orders --date-col OrderDate --customer-col CustomerID --order-col OrderNumber --amount-col Revenue
"""
import argparse
from pathlib import Path
import sys
import glob
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

# ---------------------------
# Helpers
# ---------------------------
POSSIBLE_COLS = {
    "customer_id": ["customer_id", "customer", "customerid", "cust_id", "customer_unique_id"],
    "order_id": ["order_id", "orderid", "order number", "order_no", "invoice", "invoice_no"],
    "order_date": ["order_date", "date", "orderdate", "purchase_date", "created_at"],
    "amount": ["amount", "revenue", "sales", "gmv", "total", "order_total", "price"],
    "product": ["product", "sku", "product_name", "item"]
}

def standardize_cols(df: pd.DataFrame):
    df = df.copy()
    cmap = {}
    lower = {c.lower().strip(): c for c in df.columns}
    for target, candidates in POSSIBLE_COLS.items():
        for cand in candidates:
            if cand in lower:
                cmap[lower[cand]] = target
                break
    df = df.rename(columns=cmap)
    return df

def enforce_schema(df: pd.DataFrame, args):
    df = standardize_cols(df)
    # CLI overrides
    if args.customer_col: df = df.rename(columns={args.customer_col: "customer_id"})
    if args.order_col: df = df.rename(columns={args.order_col: "order_id"})
    if args.date_col: df = df.rename(columns={args.date_col: "order_date"})
    if args.amount_col: df = df.rename(columns={args.amount_col: "amount"})

    required = ["customer_id", "order_id", "order_date", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Types
    df["customer_id"] = df["customer_id"].astype(str)
    df["order_id"] = df["order_id"].astype(str)

    # Parse dates robustly
    if not np.issubdtype(df["order_date"].dtype, np.datetime64):
        def parse_dt(x):
            if pd.isna(x): return pd.NaT
            try:
                return pd.to_datetime(x)
            except Exception:
                try:
                    return pd.to_datetime(dateparser.parse(str(x)))
                except Exception:
                    return pd.NaT
        df["order_date"] = df["order_date"].apply(parse_dt)

    # Numeric amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Clean
    df = df.dropna(subset=["customer_id", "order_id", "order_date", "amount"])
    df = df[df["amount"] >= 0]
    return df

def load_inputs(paths, sheet=None):
    files = []
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):
            files.extend(glob.glob(p))
        else:
            pth = Path(p)
            if pth.is_dir():
                files.extend([f.as_posix() for f in Path(p).glob("**/*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"]])
            else:
                files.append(p)
    if not files:
        raise FileNotFoundError("No input files found.")

    frames = []
    for f in files:
        fp = Path(f)
        if fp.suffix.lower() in [".xlsx", ".xls"]:
            if sheet:
                df = pd.read_excel(fp, sheet_name=sheet)
            else:
                # read all sheets and concat
                book = pd.read_excel(fp, sheet_name=None)
                df = pd.concat(book.values(), ignore_index=True)
        elif fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        else:
            continue
        df["_source"] = fp.name
        frames.append(df)
    if not frames:
        raise RuntimeError("No readable data loaded from inputs.")
    out = pd.concat(frames, ignore_index=True)
    return out

# ---------------------------
# Analytics
# ---------------------------
def compute_kpis(orders: pd.DataFrame):
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["order_month"] = orders["order_date"].values.astype("datetime64[M]")

    total_customers = orders["customer_id"].nunique()
    total_orders = orders["order_id"].nunique()
    total_revenue = float(orders["amount"].sum())
    aov = float(orders.groupby("order_id")["amount"].sum().mean()) if total_orders else 0.0

    orders_per_cust = orders.groupby("customer_id")["order_id"].nunique()
    repeat_rate = float((orders_per_cust > 1).mean()) if total_customers else 0.0

    gaps = orders.sort_values(["customer_id", "order_date"]).groupby("customer_id")["order_date"].diff().dt.days
    median_reorder_days = float(gaps.dropna().median()) if gaps.notna().any() else np.nan

    revenue_by_month = orders.groupby("order_month")["amount"].sum().reset_index()
    top_customers = orders.groupby("customer_id")["amount"].sum().sort_values(ascending=False).reset_index().head(20)

    return {
        "kpis": pd.DataFrame([{
            "customers": int(total_customers),
            "orders": int(total_orders),
            "revenue": round(total_revenue, 2),
            "aov": round(aov, 2),
            "repeat_purchase_rate": round(repeat_rate, 4),
            "median_reorder_days": median_reorder_days if not np.isnan(median_reorder_days) else None
        }]),
        "revenue_by_month": revenue_by_month,
        "top_customers": top_customers
    }

def rfm(orders: pd.DataFrame, quantiles=5):
    df = orders.copy()
    ref_date = pd.to_datetime(df["order_date"]).max() + pd.Timedelta(days=1)
    agg = df.groupby("customer_id").agg(
        recency_days=("order_date", lambda s: (ref_date - pd.to_datetime(s).max()).days),
        frequency=("order_id", "nunique"),
        monetary=("amount", "sum")
    ).reset_index()

    agg["monetary"] = agg["monetary"].clip(lower=0.0)

    def qscore(x, reverse=False, q=quantiles):
        # robust qcut with fallback
        try:
            labels = list(range(1, q+1))
            qv = pd.qcut(x.rank(method="first"), q, labels=labels)
        except Exception:
            labels = [1, 3, 5]
            qv = pd.qcut(x.rank(method="first"), 3, labels=labels)
        qv = qv.astype(int)
        return (q+1 - qv) if reverse else qv

    agg["R_score"] = qscore(agg["recency_days"], reverse=True)
    agg["F_score"] = qscore(agg["frequency"], reverse=False)
    agg["M_score"] = qscore(agg["monetary"], reverse=False)
    agg["RFM_score"] = agg["R_score"]*100 + agg["F_score"]*10 + agg["M_score"]

    def label_segment(r, f):
        if r >= 4 and f >= 4: return "Champions"
        if r >= 4 and f >= 2: return "Loyal"
        if r >= 3 and f >= 3: return "Potential Loyalist"
        if r <= 2 and f >= 4: return "At Risk Loyal"
        if r <= 2 and f <= 2: return "Hibernating"
        if r >= 3 and f <= 2: return "New Customers"
        return "Needs Attention"

    agg["segment"] = [label_segment(r, f) for r, f in zip(agg["R_score"], agg["F_score"])]
    seg_summary = agg.groupby("segment").agg(
        customers=("customer_id", "nunique"),
        avg_R=("R_score", "mean"),
        avg_F=("F_score", "mean"),
        avg_M=("M_score", "mean"),
        avg_monetary=("monetary", "mean")
    ).reset_index().sort_values("customers", ascending=False)

    return agg, seg_summary

def cohort_retention(orders: pd.DataFrame):
    df = orders.copy()
    df["order_month"] = pd.to_datetime(df["order_date"]).values.astype("datetime64[M]")
    first = df.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df = df.merge(first, on="customer_id", how="left")
    df["period_number"] = ((df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
                           + (df["order_month"].dt.month - df["cohort_month"].dt.month))
    cohort_sizes = df.groupby("cohort_month")["customer_id"].nunique()
    active = df.groupby(["cohort_month", "period_number"])["customer_id"].nunique().reset_index()
    retention = active.pivot(index="cohort_month", columns="period_number", values="customer_id").fillna(0)
    retention = retention.div(cohort_sizes, axis=0)
    retention = retention.reset_index()
    return retention

# ---------------------------
# Reporting (Excel + CSV)
# ---------------------------
def write_excel_report(out_path: Path, results: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        wb = writer.book

        # Summary
        results["kpis"].to_excel(writer, sheet_name="Summary", index=False, startrow=1)
        rev_month = results["revenue_by_month"].copy()
        if not rev_month.empty:
            rev_month.rename(columns={"order_month": "month", "amount": "revenue"}, inplace=True)
        rev_month.to_excel(writer, sheet_name="RevenueByMonth", index=False)
        results["top_customers"].to_excel(writer, sheet_name="TopCustomers", index=False)

        # RFM
        results["rfm"].to_excel(writer, sheet_name="RFM", index=False)
        results["segments"].to_excel(writer, sheet_name="Segments", index=False)

        # Cohorts
        cohorts = results["cohorts"].copy()
        cohorts.to_excel(writer, sheet_name="Cohorts", index=False)

        # Formats and chart on Summary
        sh = writer.sheets["Summary"]
        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        note_fmt = wb.add_format({"italic": True, "font_color": "#666666"})
        sh.write(0, 0, "Customer Behavior KPIs", title_fmt)
        sh.write(10, 0, "Sheets: RFM, Segments, Cohorts, RevenueByMonth, TopCustomers", note_fmt)

        # Segment bar chart
        seg_df = results["segments"]
        if not seg_df.empty:
            seg_sheet = writer.sheets["Segments"]
            # Place chart on Summary
            chart = wb.add_chart({"type": "column"})
            # Data range from Segments sheet
            chart.add_series({
                "name": "Customers by Segment",
                "categories": ["Segments", 1, 0, len(seg_df), 0],
                "values": ["Segments", 1, 1, len(seg_df), 1],
                "data_labels": {"value": True}
            })
            chart.set_title({"name": "Customers by Segment"})
            chart.set_x_axis({"name": "Segment"})
            chart.set_y_axis({"name": "Customers"})
            sh.insert_chart("E2", chart, {"x_scale": 1.2, "y_scale": 1.2})

        # Cohort heatmap via conditional formatting
        coh_sheet = writer.sheets["Cohorts"]
        if "Cohorts" in writer.sheets and not cohorts.empty and cohorts.shape[1] > 2:
            nrows, ncols = cohorts.shape
            # Apply 3-color scale on retention values (exclude cohort_month column)
            coh_sheet.conditional_format(1, 1, nrows, ncols-1, {
                "type": "3_color_scale",
                "min_color": "#f7fbff",
                "mid_color": "#6baed6",
                "max_color": "#08306b"
            })

def save_csv_exports(out_dir: Path, results: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    results["kpis"].to_csv(out_dir / "kpis.csv", index=False)
    results["rfm"].to_csv(out_dir / "rfm_customers.csv", index=False)
    results["segments"].to_csv(out_dir / "segment_profiles.csv", index=False)
    results["cohorts"].to_csv(out_dir / "cohort_retention.csv", index=False)
    results["revenue_by_month"].to_csv(out_dir / "revenue_by_month.csv", index=False)
    results["top_customers"].to_csv(out_dir / "top_customers.csv", index=False)

# ---------------------------
# Main
# ---------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="Customer Behavior Analytics Report")
    p.add_argument("--input", "-i", nargs="+", required=True, help="Input files/dirs/globs (Excel/CSV)")
    p.add_argument("--sheet", help="Excel sheet name (if single sheet to read)")
    p.add_argument("--out", "-o", default="outputs/customer_behavior_report.xlsx", help="Output Excel path")
    p.add_argument("--csv-out", default="outputs/customer_behavior", help="Output CSV directory")
    p.add_argument("--date-col", help="Override order date column name")
    p.add_argument("--customer-col", help="Override customer column name")
    p.add_argument("--order-col", help="Override order id column name")
    p.add_argument("--amount-col", help="Override amount/revenue column name")
    p.add_argument("--rfm-quantiles", type=int, default=5, help="Quantiles for RFM scores (default: 5)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    raw = load_inputs(args.input, sheet=args.sheet)
    raw = enforce_schema(raw, args)

    # Core analytics
    kpi_parts = compute_kpis(raw)
    rfm_df, seg_summary = rfm(raw, quantiles=args.rfm_quantiles)
    retention = cohort_retention(raw)

    results = {
        "kpis": kpi_parts["kpis"],
        "revenue_by_month": kpi_parts["revenue_by_month"],
        "top_customers": kpi_parts["top_customers"],
        "rfm": rfm_df,
        "segments": seg_summary,
        "cohorts": retention
    }

    write_excel_report(Path(args.out), results)
    save_csv_exports(Path(args.csv_out), results)

    print(f"Report written to {args.out}")
    print(f"CSVs written to {args.csv_out}/")

if __name__ == "__main__":
    main()
