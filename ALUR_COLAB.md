# Alur Lengkap Google Colab

## 1. Clone repository

```python
!git clone https://github.com/ganggawardana92-design/forecast_bass_xgboost.git
%cd forecast_bass_xgboost
```

## 2. Install dependency

```python
!pip install -q -r requirements.txt
```

Dependency:
- numpy: operasi numerik
- pandas: pengolahan tabel/time series
- scipy: estimasi parameter Bass dengan curve_fit
- scikit-learn: MAE dan RMSE
- xgboost: XGBoost standalone dan residual learner
- matplotlib: grafik
- dbfread: membaca ANALIS.xls yang sebenarnya berformat DBF

## 3. Upload data mentah

```python
from google.colab import files
uploaded = files.upload()
```

Pilih `ANALIS.xls`.

## 4. Preprocessing

```python
!python preprocess.py --input /content/ANALIS.xls
```

Preprocessing:
1. Membaca DBF.
2. Menghitung distribusi TIPE J/B/D/K.
3. Memilih TIPE J sebagai penjualan/konsumsi.
4. Mempertahankan QTY negatif J sebagai net sales dan menandainya sebagai kandidat retur/koreksi.
5. Agregasi QTY per tanggal.
6. Membentuk kalender harian lengkap.
7. Tanggal tanpa transaksi diberi nilai 0, tetapi ditandai agar dapat diaudit.
8. Outlier IQR hanya ditandai, tidak dihapus otomatis.
9. Menyimpan `data/daily_sales_jual.csv` dan `data/preprocessing_summary.csv`.

## 5. Audit preprocessing

```python
import pandas as pd

daily = pd.read_csv("data/daily_sales_jual.csv")
summary = pd.read_csv("data/preprocessing_summary.csv")

display(summary)
display(daily.head())
display(daily[daily["is_zero_day"] == 1])
display(daily[daily["is_iqr_outlier"] == 1].sort_values("sales_qty", ascending=False).head(20))
```

Jangan langsung menghapus zero/outlier sebelum memastikan apakah itu benar-benar error data atau kejadian penjualan nyata.

## 6. Jalankan model

```python
!python code.py
```

Model:
1. Split 80:20 kronologis.
2. Fit Bass pada train.
3. Hitung residual train: actual - Bass.
4. XGBoost mempelajari residual.
5. XGBoost standalone digunakan sebagai pembanding.
6. Forecast test secara recursive.
7. Hitung MAE, RMSE, MAPE.
8. Refit seluruh data.
9. Forecast 30 hari ke depan.

## 7. Baca hasil evaluasi

```python
metrics = pd.read_csv("metrik_evaluasi.csv")
eval_df = pd.read_csv("hasil_evaluasi_test.csv")

display(metrics)
display(eval_df.head())
```

Interpretasi:
- MAE makin kecil = rata-rata error absolut makin kecil.
- RMSE makin kecil = model makin sedikit membuat error besar.
- MAPE makin kecil = error persentase rata-rata makin kecil.
- Jangan menyimpulkan Hybrid terbaik sebelum melihat hasil aktual.

## 8. Baca forecast 30 hari

```python
forecast = pd.read_csv("hasil_forecast_30_hari.csv")
display(forecast)
```

Kolom:
- bass_forecast = baseline Bass.
- xgboost_residual_correction = koreksi yang diprediksi XGBoost.
- hybrid_forecast_qty = prediksi akhir Bass + koreksi residual.

Forecast ini adalah perkiraan QTY penjualan/konsumsi untuk membantu perencanaan stok, bukan stok fisik aktual.

## 9. Download output

```python
from google.colab import files

for f in [
    "data/preprocessing_summary.csv",
    "data/daily_sales_jual.csv",
    "metrik_evaluasi.csv",
    "hasil_evaluasi_test.csv",
    "hasil_forecast_30_hari.csv",
    "plot_evaluasi_test.png",
    "plot_forecast_30_hari.png",
]:
    files.download(f)
```

## 10. Analisis untuk tesis

Laporkan:
1. Struktur dan periode data.
2. Aturan pemilihan TIPE J.
3. Penanganan QTY negatif, zero days, dan outlier.
4. Parameter Bass p, q, m.
5. Residual Bass.
6. Feature XGBoost.
7. MAE/RMSE/MAPE Bass, XGBoost, Hybrid.
8. Grafik aktual vs prediksi.
9. Forecast 30 hari.
10. Implikasi forecast terhadap kebutuhan stok.

Catatan: untuk tesis, sebaiknya lakukan validasi time-series tambahan/tuning setelah pipeline dasar ini dipastikan benar.
