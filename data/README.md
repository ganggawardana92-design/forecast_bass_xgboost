# Data

`daily_sales_jual.csv` adalah data agregat harian yang dibuat dari file sumber `ANALIS.xls`.

Transformasi:
1. File sumber dibaca sebagai DBF/FoxBase (meskipun ekstensi .xls).
2. Hanya transaksi `TIPE='J'` yang dianggap penjualan/konsumsi.
3. `QTY` dijumlahkan berdasarkan tanggal.
4. Kalender harian dibuat lengkap; tanggal tanpa penjualan diisi 0.

Kolom yang dipublikasikan:
- `date`
- `sales_qty`

Data transaksi mentah tidak dipublikasikan di repo karena repository saat ini bersifat public dan data mentah memuat detail transaksi/barang. Jika memang ingin mempublikasikan file mentah, ubah repo menjadi private atau konfirmasi secara eksplisit.
