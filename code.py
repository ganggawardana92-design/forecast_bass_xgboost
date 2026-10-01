"""
Hybrid Bass Diffusion + XGBoost untuk forecasting konsumsi/penjualan supermarket.

Konteks:
- "stock" = stok/inventory supermarket, bukan financial stock market.
- Target = QTY penjualan/konsumsi harian.
- Output = forecast 30 hari untuk dukungan perencanaan stok.

Dataset repo:
data/daily_sales_jual.csv
Hanya transaksi TIPE='J' (penjualan) dari data mentah yang sudah diagregasi per hari.
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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_URL = "https://raw.githubusercontent.com/ganggawardana92-design/forecast_bass_xgboost/main/data/daily_sales_jual.csv"\nLOCAL_DATA_PATH = "data/daily_sales_jual.csv"
TRAIN_RATIO = 0.80
FORECAST_HORIZON = 30
RANDOM_STATE = 42
LAGS = [1,7,14]
MA_WINDOWS = [7,14]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

def load_data():
    # Jika preprocess.py baru saja dijalankan, gunakan hasil lokal.
    # Jika tidak ada, fallback ke data agregat yang tersimpan di GitHub.
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

def bass_cumulative(t,p,q,m):
    t=np.asarray(t,dtype=float)
    p=max(float(p),1e-10); q=max(float(q),1e-10); m=max(float(m),1e-10)
    e=np.exp(-(p+q)*t)
    return m*((1-e)/(1+(q/p)*e))

def bass_daily(t,p,q,m):
    t=np.asarray(t,dtype=float)
    return np.maximum(bass_cumulative(t+1,p,q,m)-bass_cumulative(t,p,q,m),0.0)

def fit_bass(y):
    y=np.asarray(y,dtype=float)
    cumulative=np.cumsum(np.maximum(y,0.0))
    t=np.arange(1,len(cumulative)+1,dtype=float)
    max_cum=max(float(cumulative[-1]),1.0)
    params,_=curve_fit(
        bass_cumulative,t,cumulative,
        p0=[0.01,0.10,max_cum*1.25],
        bounds=([1e-8,1e-8,max_cum],[1.0,5.0,max_cum*20.0]),
        maxfev=200000
    )
    return tuple(float(v) for v in params)

def bass_series(n,params,start_t=0):
    p,q,m=params
    t=np.arange(start_t,start_t+n,dtype=float)
    return bass_daily(t,p,q,m)

def cal_features(date):
    date=pd.Timestamp(date)
    return {
        "dayofweek":float(date.dayofweek),
        "dayofmonth":float(date.day),
        "month":float(date.month),
        "weekofyear":float(date.isocalendar().week),
        "is_weekend":float(date.dayofweek>=5),
    }

def features(history,date,time_index,bass_pred=None):
    f={}
    for lag in LAGS:
        f[f"lag_{lag}"]=float(history[-lag]) if len(history)>=lag else np.nan
    for w in MA_WINDOWS:
        f[f"ma_{w}"]=float(np.mean(history[-w:])) if len(history)>=w else np.nan
    f.update(cal_features(date))
    f["time_index"]=float(time_index)
    if bass_pred is not None:
        f["bass_pred"]=float(bass_pred)
    return f

def make_matrix(dates,y,bass_pred=None):
    rows=[]; history=[]
    for i,(d,target) in enumerate(zip(dates,y)):
        bp=None if bass_pred is None else float(bass_pred[i])
        r=features(history,d,i,bp)
        r["target"]=float(target); r["row_index"]=i
        rows.append(r); history.append(float(target))
    out=pd.DataFrame(rows)
    feat=[c for c in out.columns if c not in {"target","row_index"}]
    return out.dropna(subset=feat).reset_index(drop=True)

def recursive_predict(model,history,dates,start_idx,bass_values=None,mode="sales"):
    hist=[float(v) for v in history]
    preds=[]
    for step,d in enumerate(dates):
        bp=None if bass_values is None else float(bass_values[step])
        X=pd.DataFrame([features(hist,d,start_idx+step,bp)])
        z=float(model.predict(X)[0])
        pred=(bp+z) if mode=="residual" else z
        pred=max(pred,0.0)
        preds.append(pred); hist.append(pred)
    return np.asarray(preds)

def mape(y_true,y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    mask=np.abs(y_true)>1e-8
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100.0)

def metric_row(name,y_true,y_pred):
    return {
        "model":name,
        "MAE":float(mean_absolute_error(y_true,y_pred)),
        "RMSE":float(math.sqrt(mean_squared_error(y_true,y_pred))),
        "MAPE_percent":mape(y_true,y_pred),
    }

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

    bass_params=fit_bass(y_train)
    p,q,m=bass_params
    bass_train=bass_series(len(train),bass_params,0)
    bass_test=bass_series(len(test),bass_params,len(train))

    residual_train=y_train-bass_train
    hm=make_matrix(train["date"],y_train,bass_train)
    idx=hm["row_index"].astype(int).to_numpy()
    hm["target"]=residual_train[idx]
    Xh=hm.drop(columns=["target","row_index"]); yh=hm["target"]
    hybrid_model=XGBRegressor(**XGB_PARAMS).fit(Xh,yh)
    hybrid_test=recursive_predict(
        hybrid_model,y_train.tolist(),pd.DatetimeIndex(test["date"]),
        len(train),bass_test,"residual"
    )

    xm=make_matrix(train["date"],y_train,None)
    Xx=xm.drop(columns=["target","row_index"]); yx=xm["target"]
    xgb_model=XGBRegressor(**XGB_PARAMS).fit(Xx,yx)
    xgb_test=recursive_predict(
        xgb_model,y_train.tolist(),pd.DatetimeIndex(test["date"]),
        len(train),None,"sales"
    )

    metrics=pd.DataFrame([
        metric_row("Bass",y_test,bass_test),
        metric_row("XGBoost",y_test,xgb_test),
        metric_row("Hybrid Bass-XGBoost",y_test,hybrid_test),
    ])
    print("\nParameter Bass:")
    print(f"p={p:.8f}, q={q:.8f}, m={m:,.2f}")
    print("\nEvaluasi:")
    print(metrics.to_string(index=False))

    test_result=pd.DataFrame({
        "date":test["date"].to_numpy(),
        "actual":y_test,
        "bass":bass_test,
        "xgboost":xgb_test,
        "hybrid":hybrid_test
    })
    test_result.to_csv("hasil_evaluasi_test.csv",index=False)
    metrics.to_csv("metrik_evaluasi.csv",index=False)

    full_params=fit_bass(y)
    bass_full=bass_series(len(df),full_params,0)
    residual_full=y-bass_full
    fm=make_matrix(df["date"],y,bass_full)
    fidx=fm["row_index"].astype(int).to_numpy()
    fm["target"]=residual_full[fidx]
    Xf=fm.drop(columns=["target","row_index"]); yf=fm["target"]
    final_model=XGBRegressor(**XGB_PARAMS).fit(Xf,yf)

    future_dates=pd.date_range(df["date"].iloc[-1]+pd.Timedelta(days=1),periods=FORECAST_HORIZON,freq="D")
    bass_future=bass_series(FORECAST_HORIZON,full_params,len(df))
    hybrid_future=recursive_predict(
        final_model,y.tolist(),future_dates,len(df),bass_future,"residual"
    )

    future=pd.DataFrame({
        "date":future_dates,
        "bass_forecast":bass_future,
        "xgboost_residual_correction":hybrid_future-bass_future,
        "hybrid_forecast_qty":hybrid_future,
    })
    future.to_csv("hasil_forecast_30_hari.csv",index=False)

    plt.figure(figsize=(15,6))
    plt.plot(test_result["date"],test_result["actual"],label="Aktual",linewidth=2)
    plt.plot(test_result["date"],test_result["bass"],label="Bass")
    plt.plot(test_result["date"],test_result["xgboost"],label="XGBoost")
    plt.plot(test_result["date"],test_result["hybrid"],label="Hybrid")
    plt.title("Evaluasi Forecast pada Data Test")
    plt.xlabel("Tanggal"); plt.ylabel("QTY"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig("plot_evaluasi_test.png",dpi=150); plt.show()

    plt.figure(figsize=(15,6))
    n=min(60,len(df))
    plt.plot(df["date"].iloc[-n:],y[-n:],label="Aktual 60 hari terakhir",linewidth=2)
    plt.plot(future["date"],future["hybrid_forecast_qty"],label="Forecast Hybrid 30 hari",linewidth=2)
    plt.axvline(df["date"].iloc[-1],linestyle="--",alpha=.7)
    plt.title("Forecast 30 Hari untuk Dukungan Perencanaan Stok")
    plt.xlabel("Tanggal"); plt.ylabel("QTY"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig("plot_forecast_30_hari.png",dpi=150); plt.show()

    print("\nForecast 30 hari:")
    print(future.to_string(index=False))
    print("\nCatatan: output adalah forecast konsumsi/penjualan QTY, bukan level stok fisik aktual.")

if __name__=="__main__":
    main()
