# Laporan Proyek Machine Learning - Nikola Izzan Ar Rasheed

## Project Overview

Sistem rekomendasi film merupakan komponen penting dalam industri hiburan digital untuk meningkatkan keterlibatan pengguna dan kepuasan pelanggan. Dengan semakin melimpahnya jumlah konten film, pengguna sering kali kesulitan menemukan film yang sesuai dengan preferensi mereka. Untuk itu, pengembangan sistem rekomendasi yang efektif menjadi kebutuhan utama.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film berbasis *collaborative filtering* menggunakan dataset MovieLens 100K. Dataset ini disediakan oleh GroupLens Research dan telah digunakan secara luas dalam penelitian sistem rekomendasi. Dengan pendekatan ini, sistem akan menyarankan film berdasarkan kemiripan perilaku pengguna lain yang memiliki pola penilaian serupa.

Sebagaimana dijelaskan oleh Schneider et al. (2022), algoritma *recommender system* berbasis *matrix factorization* seperti collaborative filtering sangat cocok untuk memprediksi entri yang hilang (misalnya, rating) dalam matriks user-item yang sparsity-nya tinggi. Dataset benchmark seperti MovieLens 100K umumnya direduksi langsung menjadi matriks sparse tersebut, meskipun beberapa dataset mengandung *side information* tambahan yang sering diabaikan oleh algoritma RecSys klasik. Ini menunjukkan bahwa meskipun metode tradisional tetap relevan, potensi peningkatan akurasi prediksi bisa diperoleh jika informasi tambahan dan teknik *feature engineering* dari dunia machine learning turut dipertimbangkan dalam pengembangan sistem rekomendasi.

**Referensi:**

- F. Maxwell Harper and Joseph A. Konstan. "The MovieLens Datasets: History and Context." *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 2015. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872)

## Business Understanding

### Problem Statements

- Bagaimana cara memberikan rekomendasi film yang relevan bagi pengguna berdasarkan riwayat rating mereka?
- Bagaimana memprediksi rating yang mungkin diberikan oleh pengguna terhadap film yang belum mereka tonton?

### Goals

- Menghasilkan rekomendasi film yang relevan dan dipersonalisasi untuk pengguna tertentu.
- Memodelkan dan memprediksi rating yang akan diberikan oleh pengguna terhadap film tertentu dengan akurasi yang baik.

### Solution Approach

- Mengimplementasikan algoritma **User-Based Collaborative Filtering** dengan menggunakan *cosine similarity* antar pengguna.
- Membangun matriks user-item sebagai dasar sistem rekomendasi dan prediksi rating.

## Data Understanding

Dataset yang digunakan adalah **MovieLens 100K**, yang terdiri dari 100.000 rating dari 943 pengguna terhadap 1.682 film. Dataset ini sudah lumayan bersih dari awal dengan tidak adanya data duplicate dan missing value. Dataset ini tersedia secara publik dan dapat diakses melalui tautan berikut: [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/).

Dataset yang digunakan terdiri dari dua file, yaitu:

- `u.data`: data rating dengan kolom `user_id`, `item_id`, `rating`, dan `timestamp`.
- `u.item`: metadata film dengan kolom `item_id`, `title`, `release_date`, `video_release_date`, `IMDB_URL`, `unknown`, dan genre (multi-label).

Penjelasan fitur/kolom dalam dataset:
- **user_id**: ID unik pengguna.
- **item_id**: ID unik film.
- **rating**: Rating film dari rentang 1 sampai 5.
- **timestamp**: Waktu rating diberikan, dalam format UNIX timestamp.
- **title**: Judul film.
- **release_date**: Tanggal rilis film.
- **video_release_data**: Tanggal rilis versi video (VHS/DVD), meskipun sebagian besar nilainya kosong.
- **IMDB_URL**: Tautan ke halaman film tersebut di IMDb.
- **unknown**: Genre default untuk film dengan genre yang tidak diklasifikasikan secara spesifik.
- **genre (Action, Comedy, Romance, dll)**: Informasi genre film dalam bentuk biner.

**Exploratory Data Analysis**

- Visualisasi distribusi rating
![Visualisasi Distribusi Rating](https://i.imgur.com/WlMgsDI.png)
Visualisasi distribusi rating menunjukkan bahwa sebagian besar pengguna memberikan rating antara 3 dan 4. Ini mengindikasikan bahwa pengguna cenderung memberikan penilaian yang netral hingga positif terhadap film yang mereka tonton. Distribusi ini juga menunjukkan bahwa rating ekstrem (1 atau 5) relatif lebih jarang diberikan, yang bisa mengindikasikan kecenderungan pengguna untuk menghindari penilaian sangat rendah atau sangat tinggi.

- Visualisasi top 10 film
![Visualisasi top 10 film](https://i.imgur.com/NVafLjW.png)
Top 10 film dengan jumlah rating terbanyak juga ditampilkan dalam bentuk visualisasi bar chart. Film-film tersebut  merupakan film populer atau klasik yang dikenal luas oleh pengguna, sehingga lebih banyak mendapat perhatian dan dinilai oleh banyak orang. Hal ini bisa memengaruhi sistem rekomendasi karena film dengan jumlah rating lebih banyak memiliki peluang lebih tinggi untuk direkomendasikan, terutama dalam pendekatan berbasis *popularity* atau collaborative filtering.

## Data Preparation

Langkah-langkah preprocessing data meliputi:

1. Menggabungkan file u.data dan u.item berdasarkan item_id

Tujuan: File u.data berisi informasi tentang rating yang diberikan pengguna terhadap film, sedangkan file u.item berisi metadata film, seperti judul dan genre.
Langkah: Data dari kedua file digabungkan berdasarkan kolom item_id sehingga setiap baris dalam dataset mencerminkan informasi lengkap—siapa pengguna, film apa yang dinilai, nilai rating, serta judul film.
Hasil: Dataset hasil penggabungan ini memudahkan proses analisis dan pembuatan matriks user-item.

2. Mengecek dan memastikan tidak ada nilai yang hilang dan duplikat

Tujuan: Data yang bersih sangat penting untuk menghindari bias dan kesalahan dalam proses training model.
Langkah:Dicek apakah ada nilai yang kosong (missing values) dalam kolom-kolom penting seperti user_id, item_id, rating, atau title.
Diperiksa juga apakah ada baris data yang duplikat, yaitu baris dengan semua nilai identik yang muncul lebih dari satu kali.
Hasil: Ditemukan bahwa dataset tidak memiliki nilai kosong maupun data duplikat, sehingga dapat langsung digunakan.

3. Membuat user-item matrix

Tujuan: Collaborative filtering berbasis user-user memerlukan representasi dalam bentuk matriks, di mana setiap baris mewakili seorang pengguna dan setiap kolom mewakili sebuah film.
Langkah: Mengubah data rating menjadi format matriks menggunakan fungsi pivot_table dari pandas.
Pada matriks tersebut, sel-sel berisi nilai rating yang diberikan oleh pengguna terhadap film tertentu. Jika pengguna belum menilai film, maka sel tersebut akan kosong (NaN).
Hasil: Terbentuk sebuah user-item matrix yang digunakan sebagai input utama dalam perhitungan kemiripan antar pengguna (user-user similarity).

4. Mengisi nilai kosong dengan 0

Tujuan: Algoritma seperti cosine similarity tidak dapat menghitung kemiripan jika terdapat nilai kosong (NaN) dalam data. Oleh karena itu, matriks harus penuh.
Langkah: Seluruh nilai kosong pada user-item matrix diisi dengan angka 0.
Asumsi yang digunakan adalah bahwa ketiadaan rating dianggap sebagai “tidak ada interaksi”.
Hasil: Matriks menjadi siap digunakan untuk perhitungan kemiripan dengan metode cosine similarity.

Langkah ini penting untuk memastikan data dalam format yang tepat sebelum digunakan dalam pemodelan.

## Modeling

Model yang digunakan adalah **User-Based Collaborative Filtering**:

- Menghitung kemiripan antar pengguna menggunakan *cosine similarity*.
- Untuk menghasilkan rekomendasi, sistem mencari pengguna lain yang paling mirip, dan mengambil film yang disukai oleh mereka untuk direkomendasikan ke pengguna target.

Output model berupa rekomendasi top-N film (misalnya 10 film) yang belum ditonton oleh pengguna.
Contoh rekomendasi untuk user ID 40 dan 51 berhasil ditampilkan berdasarkan pendekatan ini.

Berikut merupakan contoh prediksi oleh user ID 40:
**Prediksi rekomendasi**: 
| Judul Film                          | Skor Rekomendasi            |
|------------------------------------|-----------------|
| Apt Pupil (1998)                   | 15.408980       |
| Titanic (1997)                     | 14.580566       |
| Apostle, The (1997)                | 12.512371       |
| In & Out (1997)                    | 12.356969       |
| Devil's Advocate, The (1997)       | 11.231197       |
| Seven Years in Tibet (1997)        | 10.675532       |
| Everyone Says I Love You (1996)    | 10.651145       |
| Scream (1996)                      | 10.529645       |
| Cop Land (1997)                    | 9.948980        |
| In the Company of Men (1997)       | 8.710036        |

**Prediksi rating terhadap film *Titanic (1997)***:
| Rating             |
|------------------------|
| 3.614123152617995      |

Berikut merupakan contoh prediksi oleh user ID 51:
**Prediksi rekomendasi**: 
| Judul Film                                | Skor Rekomendasi  |
|-------------------------------------------|-------------|
| Raiders of the Lost Ark (1981)            | 14.144349 |
| Terminator 2: Judgment Day (1991)         | 11.307355 |
| Monty Python and the Holy Grail (1974)    | 10.940803 |
| Toy Story (1995)                          | 10.638266 |
| Aliens (1986)                             | 9.740786 |
| Back to the Future (1985)                 | 9.677367 |
| Casablanca (1942)                         | 9.065003 |
| Braveheart (1995)                         | 8.729574 |
| Schindler's List (1993)                   | 8.692218 |
| Alien (1979)                              | 8.470288 |


**Prediksi rating terhadap film *Star Wars (1977)***:
| Rating             |
|------------------------|
| 4.93      |

Kelebihan:
- Sederhana dan mudah diimplementasikan: Tidak memerlukan proses training atau parameter tuning yang rumit.
- Interpretable: Hasil rekomendasi mudah dijelaskan karena berdasarkan kesamaan perilaku pengguna.
- Adaptif: Sistem langsung memperbarui rekomendasi jika ada data rating baru yang masuk.

Kekurangan:
- Masalah skala (scalability): Ketika jumlah pengguna sangat besar, menghitung kemiripan antar semua pengguna menjadi sangat mahal secara komputasi.
- Data sparsity: Pada dataset yang jarang (banyak nilai kosong), sulit menemukan pengguna yang benar-benar mirip karena sedikitnya item yang dinilai bersama.
- Cold start problem: Tidak dapat memberikan rekomendasi kepada pengguna baru (tanpa riwayat rating) atau untuk film baru yang belum memiliki rating.
- Overgeneralization: Film yang sangat populer dan banyak ditonton cenderung direkomendasikan ke semua pengguna, meskipun belum tentu sesuai preferensi individu.

## Evaluation

Model dievaluasi menggunakan **RMSE** dan **MSE** terhadap 1000 sampel rating acak dari dataset.

**Langkah Evaluasi**:

1. Mengambil 1000 sampel acak dari data asli.
2. Menghitung prediksi rating untuk setiap sampel.
3. Menggunakan fungsi `mean_squared_error` dari `sklearn.metrics` untuk menghitung MSE dan RMSE.

**Hasil Evaluasi:**

- Mean Squared Error (MSE): 0.9121
- Root Mean Squared Error (RMSE): 0.9551

Metrik RMSE digunakan karena sesuai untuk masalah prediksi nilai kontinu seperti rating film. Semakin kecil nilai RMSE, semakin baik prediksi yang dihasilkan oleh model.

**Formula Metrik**:

![Formula MSE](https://github.com/user-attachments/assets/4f3ecf20-68e0-483b-9afb-9add57074433)

MSE menghitung rata-rata kuadrat selisih antara nilai aktual dan prediksi. Nilai MSE yang lebih kecil menunjukkan prediksi yang lebih akurat.


![Formula RMSE](https://github.com/user-attachments/assets/19e779e6-5238-4c15-a475-759600ad708d)

RMSE adalah akar kuadrat dari MSE, dan memiliki satuan yang sama dengan target variabel 𝑦
RMSE lebih sensitif terhadap outlier dibanding MAE, karena penalti kuadrat.

- yᵢ : nilai aktual ke-i
- ŷᵢ : nilai prediksi ke-i
- Ȳ : rata-rata dari seluruh nilai aktual
- n : jumlah total data
---
