"""
Forecast V2 — Hybrid Bass Diffusion + XGBoost Residual Correction (lebih robust).

Perbaikan dibanding forecast.py:
1. Fitur lag lebih lengkap: 1,2,3,7,14,21,28.
2. Rolling mean, median, dan std.
3. Fitur kalender + seasonality tahunan siklik.
4. XGBoost memakai pseudo-Huber loss agar lebih tahan terhadap lonjakan ekstrem.
5. Residual correction dibatasi berdasarkan distribusi residual training.
6. Besar koreksi dikendalikan oleh alpha:
       hybrid = bass + alpha * predicted_residual
   Alpha dipilih dari VALIDATION secara kronologis, bukan dari test.
7. Tetap memakai test 20% terakhir untuk evaluasi final yang fair.
"""

import os, sys, math, subprocess, warnings
warnings.filterwarnings("ignore")

def ensure(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])

for imp, pipn in [
    ("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),
    ("sklearn","scikit-learn"),("xgboost","xgboost"),("matplotlib","matplotlib")
]:
    ensure(imp,pipn)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_URL = "https://raw.githubusercontent.com/ganggawardana92-design/forecast_bass_xgboost/main/data/daily_sales_jual.csv"
LOCAL_DATA_PATH = "data/daily_sales_jual.csv"

TRAIN_RATIO = 0.80
INNER_TRAIN_RATIO = 0.85
FORECAST_HORIZON = 30
RANDOM_STATE = 42

LAGS = [1,2,3,7,14,21,28]
ROLL_WINDOWS = [7,14,28]
ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)

XGB_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.10,
    "reg_lambda": 2.0,
    "objective": "reg:pseudohubererror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

def load_data():
    source = LOCAL_DATA_PATH if os.path.exists(LOCAL_DATA_PATH) else DATA_URL
    print(f"Data model: {source}")
    df = pd.read_csv(source)
    required = {"date", "sales_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0.0)
    df = df.sort_values("date").drop_duplicates("date", keep="last")

    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates)
    df.index.name = "date"
    df["sales_qty"] = df["sales_qty"].fillna(0.0)
    return df.reset_index()

# ---------------- BASS ----------------

def bass_cumulative(t,p,q,m):
    t=np.asarray(t,dtype=float)
    p=max(float(p),1e-10)
    q=max(float(q),1e-10)
    m=max(float(m),1e-10)
    e=np.exp(-(p+q)*t)
    return m*((1-e)/(1+(q/p)*e))

def bass_daily(t,p,q,m):
    t=np.asarray(t,dtype=float)
    return np.maximum(
        bass_cumulative(t+1,p,q,m)-bass_cumulative(t,p,q,m),
        0.0
    )

def fit_bass(y):
    y=np.asarray(y,dtype=float)
    cumulative=np.cumsum(np.maximum(y,0.0))
    t=np.arange(1,len(cumulative)+1,dtype=float)
    max_cum=max(float(cumulative[-1]),1.0)

    params,_=curve_fit(
        bass_cumulative,
        t,
        cumulative,
        p0=[0.01,0.10,max_cum*1.25],
        bounds=([1e-8,1e-8,max_cum],[1.0,5.0,max_cum*20.0]),
        maxfev=200000
    )
    return tuple(float(v) for v in params)

def bass_series(n,params,start_t=0):
    p,q,m=params
    t=np.arange(start_t,start_t+n,dtype=float)
    return bass_daily(t,p,q,m)

# ---------------- FEATURES ----------------

def calendar_features(date):
    d=pd.Timestamp(date)
    doy=float(d.dayofyear)
    dow=float(d.dayofweek)

    return {
        "dayofweek": dow,
        "dayofmonth": float(d.day),
        "month": float(d.month),
        "weekofyear": float(d.isocalendar().week),
        "dayofyear": doy,
        "is_weekend": float(d.dayofweek>=5),
        "is_month_start": float(d.is_month_start),
        "is_month_end": float(d.is_month_end),
        # Cyclical encoding: membantu pola mingguan/tahunan.
        "dow_sin": float(np.sin(2*np.pi*dow/7.0)),
        "dow_cos": float(np.cos(2*np.pi*dow/7.0)),
        "doy_sin": float(np.sin(2*np.pi*doy/365.25)),
        "doy_cos": float(np.cos(2*np.pi*doy/365.25)),
    }

def feature_row(history,date,time_index,bass_pred=None):
    hist=np.asarray(history,dtype=float)
    f={}

    for lag in LAGS:
        f[f"lag_{lag}"]=float(hist[-lag]) if len(hist)>=lag else np.nan

    for w in ROLL_WINDOWS:
        if len(hist)>=w:
            x=hist[-w:]
            f[f"mean_{w}"]=float(np.mean(x))
            f[f"median_{w}"]=float(np.median(x))
            f[f"std_{w}"]=float(np.std(x,ddof=0))
        else:
            f[f"mean_{w}"]=np.nan
            f[f"median_{w}"]=np.nan
            f[f"std_{w}"]=np.nan

    f.update(calendar_features(date))
    f["time_index"]=float(time_index)

    if bass_pred is not None:
        f["bass_pred"]=float(bass_pred)

    return f

def make_matrix(dates,y,bass_pred=None):
    rows=[]
    history=[]

    for i,(d,target) in enumerate(zip(dates,y)):
        bp=None if bass_pred is None else float(bass_pred[i])
        r=feature_row(history,d,i,bp)
        r["target"]=float(target)
        r["row_index"]=i
        rows.append(r)
        history.append(float(target))

    out=pd.DataFrame(rows)
    feature_cols=[c for c in out.columns if c not in {"target","row_index"}]
    return out.dropna(subset=feature_cols).reset_index(drop=True)

# ---------------- PREDICTION ----------------

def correction_bounds(residuals):
    """
    Batasi koreksi XGBoost menggunakan quantile residual TRAINING.
    Ini tidak menghapus outlier aktual, hanya mencegah correction menjadi terlalu liar.
    """
    r=np.asarray(residuals,dtype=float)
    low=float(np.quantile(r,0.02))
    high=float(np.quantile(r,0.98))
    return low,high

def recursive_xgb_sales(model,history,dates,start_idx):
    hist=[float(v) for v in history]
    preds=[]

    for step,d in enumerate(dates):
        X=pd.DataFrame([feature_row(hist,d,start_idx+step,None)])
        pred=max(float(model.predict(X)[0]),0.0)
        preds.append(pred)
        hist.append(pred)

    return np.asarray(preds)

def recursive_hybrid(model,history,dates,start_idx,bass_values,alpha,
                     corr_low=None,corr_high=None):
    hist=[float(v) for v in history]
    preds=[]
    corrections=[]

    for step,d in enumerate(dates):
        bp=float(bass_values[step])
        X=pd.DataFrame([feature_row(hist,d,start_idx+step,bp)])
        corr=float(model.predict(X)[0])

        if corr_low is not None and corr_high is not None:
            corr=float(np.clip(corr,corr_low,corr_high))

        pred=max(bp + float(alpha)*corr, 0.0)

        preds.append(pred)
        corrections.append(corr)
        hist.append(pred)

    return np.asarray(preds), np.asarray(corrections)

# ---------------- METRICS ----------------

def mape(y_true,y_pred):
    y_true=np.asarray(y_true,float)
    y_pred=np.asarray(y_pred,float)
    mask=np.abs(y_true)>1e-8

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100.0
    )

def metric_row(name,y_true,y_pred):
    return {
        "model":name,
        "MAE":float(mean_absolute_error(y_true,y_pred)),
        "RMSE":float(math.sqrt(mean_squared_error(y_true,y_pred))),
        "MAPE_percent":mape(y_true,y_pred),
    }

# ---------------- ALPHA VALIDATION ----------------

def choose_alpha(train_dates,y_train):
    """
    Alpha dipilih hanya dari bagian training:
    inner train -> validation kronologis.
    Test final sama sekali tidak dipakai memilih alpha.
    """
    inner_split=int(len(y_train)*INNER_TRAIN_RATIO)

    d_fit=train_dates.iloc[:inner_split]
    d_val=train_dates.iloc[inner_split:]

    y_fit=np.asarray(y_train[:inner_split],dtype=float)
    y_val=np.asarray(y_train[inner_split:],dtype=float)

    bass_params=fit_bass(y_fit)
    bass_fit=bass_series(len(y_fit),bass_params,0)
    bass_val=bass_series(len(y_val),bass_params,len(y_fit))

    residual_fit=y_fit-bass_fit
    mat=make_matrix(d_fit,y_fit,bass_fit)
    idx=mat["row_index"].astype(int).to_numpy()
    mat["target"]=residual_fit[idx]

    X=mat.drop(columns=["target","row_index"])
    target=mat["target"]

    model=XGBRegressor(**XGB_PARAMS).fit(X,target)

    lo,hi=correction_bounds(residual_fit)

    rows=[]
    for alpha in ALPHA_GRID:
        pred,_=recursive_hybrid(
            model,
            y_fit.tolist(),
            pd.DatetimeIndex(d_val),
            len(y_fit),
            bass_val,
            alpha=float(alpha),
            corr_low=lo,
            corr_high=hi,
        )
        rows.append(metric_row(f"alpha_{alpha:.1f}",y_val,pred) | {"alpha":float(alpha)})

    table=pd.DataFrame(rows).sort_values(["MAE","RMSE"]).reset_index(drop=True)
    best_alpha=float(table.iloc[0]["alpha"])

    table.to_csv("validasi_alpha.csv",index=False)

    print("\nValidasi alpha (5 terbaik berdasarkan MAE):")
    print(table.head(5).to_string(index=False))
    print(f"\nAlpha terpilih = {best_alpha:.2f}")

    return best_alpha

# ---------------- MAIN ----------------

def main():
    df=load_data()
    y=df["sales_qty"].to_numpy(float)

    split=int(len(df)*TRAIN_RATIO)
    train=df.iloc[:split].copy()
    test=df.iloc[split:].copy()

    y_train=train["sales_qty"].to_numpy(float)
    y_test=test["sales_qty"].to_numpy(float)

    print("Periode:",df["date"].min().date(),"s.d.",df["date"].max().date())
    print("Train/Test:",len(train),len(test))

    # 1. Pilih alpha menggunakan validation di dalam TRAIN.
    best_alpha=choose_alpha(train["date"],y_train)

    # 2. Fit Bass pada TRAIN penuh.
    bass_params=fit_bass(y_train)
    p,q,m=bass_params

    bass_train=bass_series(len(train),bass_params,0)
    bass_test=bass_series(len(test),bass_params,len(train))

    # 3. Residual Bass -> XGBoost.
    residual_train=y_train-bass_train

    hm=make_matrix(train["date"],y_train,bass_train)
    idx=hm["row_index"].astype(int).to_numpy()
    hm["target"]=residual_train[idx]

    Xh=hm.drop(columns=["target","row_index"])
    yh=hm["target"]

    hybrid_model=XGBRegressor(**XGB_PARAMS).fit(Xh,yh)

    lo,hi=correction_bounds(residual_train)

    # raw alpha=1 untuk pembanding
    hybrid_raw, corr_raw = recursive_hybrid(
        hybrid_model,
        y_train.tolist(),
        pd.DatetimeIndex(test["date"]),
        len(train),
        bass_test,
        alpha=1.0,
        corr_low=lo,
        corr_high=hi
    )

    # tuned shrinkage
    hybrid_tuned, corr_tuned = recursive_hybrid(
        hybrid_model,
        y_train.tolist(),
        pd.DatetimeIndex(test["date"]),
        len(train),
        bass_test,
        alpha=best_alpha,
        corr_low=lo,
        corr_high=hi
    )

    # 4. XGBoost standalone sebagai pembanding.
    xm=make_matrix(train["date"],y_train,None)
    Xx=xm.drop(columns=["target","row_index"])
    yx=xm["target"]

    xgb_model=XGBRegressor(**XGB_PARAMS).fit(Xx,yx)

    xgb_test=recursive_xgb_sales(
        xgb_model,
        y_train.tolist(),
        pd.DatetimeIndex(test["date"]),
        len(train)
    )

    # 5. Evaluasi final.
    metrics=pd.DataFrame([
        metric_row("Bass",y_test,bass_test),
        metric_row("XGBoost",y_test,xgb_test),
        metric_row("Hybrid raw alpha=1",y_test,hybrid_raw),
        metric_row(f"Hybrid tuned alpha={best_alpha:.2f}",y_test,hybrid_tuned),
    ])

    print("\nParameter Bass:")
    print(f"p={p:.8f}, q={q:.8f}, m={m:,.2f}")

    print("\nBatas residual correction dari training:")
    print(f"low={lo:.3f}, high={hi:.3f}")

    print("\nEvaluasi FINAL TEST:")
    print(metrics.to_string(index=False))

    metrics.to_csv("metrik_evaluasi_v2.csv",index=False)

    test_result=pd.DataFrame({
        "date":test["date"].to_numpy(),
        "actual":y_test,
        "bass":bass_test,
        "xgboost":xgb_test,
        "hybrid_raw":hybrid_raw,
        "hybrid_tuned":hybrid_tuned,
        "predicted_residual_raw":corr_raw,
        "predicted_residual_tuned":corr_tuned,
        "alpha":best_alpha,
    })

    test_result.to_csv("hasil_evaluasi_test_v2.csv",index=False)

    # Feature importance
    importance=pd.DataFrame({
        "feature":Xh.columns,
        "importance":hybrid_model.feature_importances_
    }).sort_values("importance",ascending=False)

    importance.to_csv("feature_importance_v2.csv",index=False)

    print("\n10 feature terpenting XGBoost residual:")
    print(importance.head(10).to_string(index=False))

    # 6. Fit final model pada seluruh data.
    full_params=fit_bass(y)
    bass_full=bass_series(len(df),full_params,0)
    residual_full=y-bass_full

    fm=make_matrix(df["date"],y,bass_full)
    fidx=fm["row_index"].astype(int).to_numpy()
    fm["target"]=residual_full[fidx]

    Xf=fm.drop(columns=["target","row_index"])
    yf=fm["target"]

    final_model=XGBRegressor(**XGB_PARAMS).fit(Xf,yf)

    full_lo,full_hi=correction_bounds(residual_full)

    future_dates=pd.date_range(
        df["date"].iloc[-1]+pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    bass_future=bass_series(
        FORECAST_HORIZON,
        full_params,
        len(df)
    )

    hybrid_future,corr_future=recursive_hybrid(
        final_model,
        y.tolist(),
        future_dates,
        len(df),
        bass_future,
        alpha=best_alpha,
        corr_low=full_lo,
        corr_high=full_hi
    )

    future=pd.DataFrame({
        "date":future_dates,
        "bass_forecast":bass_future,
        "predicted_residual":corr_future,
        "alpha":best_alpha,
        "applied_correction":best_alpha*corr_future,
        "hybrid_forecast_qty":hybrid_future,
    })

    future.to_csv("hasil_forecast_30_hari_v2.csv",index=False)

    # 7. Plot evaluasi.
    plt.figure(figsize=(15,6))
    plt.plot(test_result["date"],test_result["actual"],label="Aktual",linewidth=2)
    plt.plot(test_result["date"],test_result["bass"],label="Bass")
    plt.plot(test_result["date"],test_result["hybrid_tuned"],label=f"Hybrid tuned α={best_alpha:.2f}")
    plt.title("Evaluasi Final Test — Bass vs Hybrid V2")
    plt.xlabel("Tanggal")
    plt.ylabel("QTY")
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig("plot_evaluasi_test_v2.png",dpi=150)
    plt.close()

    # 8. Plot forecast.
    plt.figure(figsize=(15,6))
    n=min(60,len(df))
    plt.plot(
        df["date"].iloc[-n:],
        y[-n:],
        label="Aktual 60 hari terakhir",
        linewidth=2
    )
    plt.plot(
        future["date"],
        future["hybrid_forecast_qty"],
        label=f"Hybrid V2 30 hari α={best_alpha:.2f}",
        linewidth=2
    )
    plt.plot(
        future["date"],
        future["bass_forecast"],
        label="Bass baseline",
        linestyle="--"
    )
    plt.axvline(df["date"].iloc[-1],linestyle="--",alpha=.7)
    plt.title("Forecast 30 Hari — Hybrid Bass-XGBoost V2")
    plt.xlabel("Tanggal")
    plt.ylabel("QTY")
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig("plot_forecast_30_hari_v2.png",dpi=150)
    plt.close()

    print("\nForecast 30 hari V2:")
    print(future.to_string(index=False))

    print("\nFile output:")
    print("- validasi_alpha.csv")
    print("- metrik_evaluasi_v2.csv")
    print("- hasil_evaluasi_test_v2.csv")
    print("- feature_importance_v2.csv")
    print("- hasil_forecast_30_hari_v2.csv")
    print("- plot_evaluasi_test_v2.png")
    print("- plot_forecast_30_hari_v2.png")

    print("\nCatatan:")
    print("- Hybrid V2 = Bass + alpha * residual XGBoost.")
    print("- Alpha dipilih dari validation kronologis, bukan test.")
    print("- Outlier aktual tidak dihapus; yang dibatasi adalah besar correction XGBoost.")
    print("- Forecast adalah konsumsi/penjualan QTY untuk dukungan perencanaan stok.")

if __name__=="__main__":
    main()
