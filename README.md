# Forecast BASS + XGBoost untuk Stok Supermarket

Repositori ini berisi pipeline forecasting untuk penelitian tesis **Hybrid Bass Diffusion + XGBoost**.

> Dalam konteks penelitian ini, **stock** berarti stok/inventory barang supermarket, bukan financial stock market.

## Dataset

Sumber asli adalah \`ANALIS.xls\`. File tersebut ternyata berformat DBF/FoxBase walaupun berekstensi \`.xls\`.

Kode \`TIPE\` yang ditemukan pada data:
- \`J\` = penjualan
- \`B\` = pembelian
- \`D\` = mutasi masuk/debet; mayoritas \`PINDAHAN\`, sebagian \`ADJUST\`
- \`K\` = mutasi keluar/kredit; pada data yang diperiksa merupakan \`PINDAHAN\`

Untuk target forecasting konsumsi/penjualan, hanya transaksi \`TIPE='J'\` yang digunakan.

File \`data/daily_sales_jual.csv\` adalah hasil agregasi harian transaksi penjualan. Data yang dipublikasikan di repo hanya berisi tanggal dan total QTY harian, bukan detail transaksi mentah.

## Metodologi

1. Data dibagi 80% training dan 20% testing secara kronologis.
2. Bass Diffusion Model menghasilkan baseline.
3. Residual dihitung: \`e_t = Y_t - Yhat_Bass,t\`.
4. XGBoost mempelajari residual menggunakan lag, moving average, fitur kalender, time index, dan prediksi Bass.
5. Forecast hybrid: \`Yhat_Hybrid,t = Yhat_Bass,t + ehat_t\`.
6. Evaluasi: MAE, RMSE, MAPE.
7. Refit seluruh data lalu forecast 30 hari ke depan.

## Menjalankan di Google Colab

\`\`\`python
!git clone https://github.com/ganggawardana92-design/forecast_bass_xgboost.git
%cd forecast_bass_xgboost
!python forecast.py
\`\`\`

Output:
- \`hasil_evaluasi_test.csv\`
- \`metrik_evaluasi.csv\`
- \`hasil_forecast_30_hari.csv\`
- \`plot_evaluasi_test.png\`
- \`plot_forecast_30_hari.png\`

Forecast merepresentasikan perkiraan kebutuhan/konsumsi QTY untuk mendukung perencanaan stok, bukan level stok fisik aktual.

## Catatan metodologis

Bass Diffusion secara klasik dikembangkan untuk pola adopsi/difusi dengan potensi pasar terbatas. Pada penelitian ini Bass digunakan sesuai rancangan tesis sebagai baseline teoritis pada konsumsi agregat. Performa Hybrid tidak diasumsikan pasti lebih baik; hasil ditentukan dari evaluasi empiris.
