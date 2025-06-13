# MindTrack - Machine Learning / Backend

##  Deskripsi Proyek

Bagian ini mencakup pengembangan model Machine Learning untuk memprediksi status kesehatan mental dari teks bebas pengguna dan implementasi sistem rekomendasi perawatan sederhana berdasarkan prediksi tersebut. Proyek ini berfungsi sebagai komponen backend analitik yang dapat diintegrasikan dengan aplikasi frontend seperti MindTrack.

## Dataset

Dataset yang digunakan untuk melatih model adalah **"Sentiment Analysis for Mental Health"** yang tersedia di Kaggle. Dataset ini berisi pasangan pernyataan teks dan label status kesehatan mental yang terkait (misalnya, Anxiety, Depression, Normal, dll.).

*   **Sumber:** [https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
*   **File Utama:** `Combined Data.csv`
*   **Data Rekomendasi:** `mentalhealthtreatment.csv` (asumsi ini adalah file yang Anda gunakan untuk rekomendasi)

##  Persyaratan (Requirements)

Untuk menjalankan notebook pelatihan atau code inferensi di lingkungan Python, Anda memerlukan library berikut:

*   `tensorflow`
*   `keras`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn`
*   `nltk`
*   `emoji`
*   `kagglehub`
*   `tensorflowjs` (untuk konversi model)
*   `pickle` (untuk menyimpan objek Python)

Anda dapat menginstal sebagian besar dengan pip:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn nltk emoji kagglehub tensorflowjs
```
Berikut adalah langkah-langkah utama yang dilakukan dalam proyek ini dari sisi ML untuk melatih model dan menyiapkan aset untuk deployment:

Memuat Dataset:
- Menggunakan kagglehub untuk mengunduh dataset.
- Memuat Combined Data.csv ke dalam pandas DataFrame.
- Memuat mentalhealthtreatment.csv untuk sistem rekomendasi.

Prapemrosesan Data Teks:
- Menghapus baris dengan nilai NaN di kolom teks (statement).
- Melakukan pembersihan teks:
- Case folding (mengubah teks menjadi huruf kecil).
- Menghapus spasi ekstra dan leading/trailing space.
- Menghapus emoji.
- Menghapus angka.
- Menghapus tanda baca.
- Melakukan Stopword Removal (menggunakan daftar stopwords Bahasa Inggris NLTK).
- Melakukan Lemmatization (menggunakan WordNetLemmatizer NLTK).
- Membuat kolom baru dengan teks yang sudah dibersihkan.
  
Tokenisasi dan Sequencing:
- Menginisialisasi tf.keras.preprocessing.text.Tokenizer dan mem-fit-nya pada data teks yang sudah dibersihkan untuk membangun kosakata (word_index).
- Mengonversi teks menjadi urutan integer menggunakan tokenizer.
- Melakukan padding sequence agar memiliki panjang seragam (max_length).
  
Label Encoding:
- Menggunakan sklearn.preprocessing.LabelEncoder untuk mengonversi label status kesehatan mental (string) menjadi integer.
- Mengonversi label integer menjadi format one-hot encoding menggunakan tf.keras.utils.to_categorical.

Memuat GloVe Embeddings:
- Mengunduh file GloVe pre-trained embeddings (misalnya glove.6B.100d.txt).
- Memuat embeddings ke dalam dictionary (embeddings_index).

Membuat Embedding Matrix:

Membuat matriks NumPy yang berisi vektor GloVe untuk setiap kata dalam kosakata tokenizer Anda. Kata-kata yang tidak ditemukan dalam GloVe diinisialisasi dengan vektor nol.
  
Membangun dan Melatih Model BiLSTM:
- Mendefinisikan arsitektur model Sequential TensorFlow/Keras:
- Embedding layer menggunakan embedding_matrix_glove dan diatur trainable=False.
- Lapisan Bi-directional LSTM (dengan Dropout).
- Dense layers (dengan aktivasi ReLU dan Dropout).
- Output Dense layer dengan aktivasi softmax untuk klasifikasi multi-kelas.
- Mengkompilasi model (misalnya dengan Adam optimizer dan categorical crossentropy loss).
- Menghitung class weights untuk menangani ketidakseimbangan data.
- Melatih model menggunakan data teks yang sudah di-padding dan label one-hot encoded.
- Menggunakan callbacks EarlyStopping dan ModelCheckpoint untuk memantau validasi loss dan menyimpan model terbaik.
  
Evaluasi Model:

Mengevaluasi performa model yang sudah dilatih pada test set menggunakan metrik seperti akurasi dan menghasilkan Classification Report.

Sistem Rekomendasi Sederhana:

Mengembangkan fungsi Python untuk mencari dan mengembalikan daftar rekomendasi perawatan dari mentalhealthtreatment.csv berdasarkan status kesehatan mental yang diberikan (hasil prediksi model).

Penyimpanan Aset Model:
- Menyimpan model Keras yang sudah dilatih (versi terbaik dari ModelCheckpoint) ke file .keras.
- Menyimpan objek tokenizer dan label_encoder menggunakan pickle.
- Menyimpan word_index dari tokenizer dan classes_ dari label encoder ke file JSON (word_index.json, classes.json) untuk kemudahan penggunaan di lingkungan JavaScript.
  
Persiapan untuk Deployment (TensorFlow.js):
- Menggunakan tensorflowjs_converter untuk mengonversi file model .keras ke format TensorFlow.js (menghasilkan model.json dan file-file weight biner).
Inference (Menggunakan Model yang Disimpan) Anda dapat memuat model, tokenizer, dan label encoder yang sudah disimpan di lingkungan Python lain untuk melakukan prediksi pada teks input baru.
- Load model menggunakan tf.keras.models.load_model().
- Load tokenizer dan label encoder menggunakan pickle.load().
- Gunakan fungsi preprocessing teks yang sama persis seperti saat pelatihan.
- Tokenisasi dan padding teks input menggunakan tokenizer yang dimuat dan max_length yang sama.
- Lakukan prediksi menggunakan model yang dimuat.
- Konversi hasil prediksi (indeks) kembali ke nama status menggunakan label encoder yang dimuat.

Aset yang disiapkan (model TensorFlow.js, word_index.json, classes.json) dapat diintegrasikan ke backend Node.js menggunakan library @tensorflow/tfjs-node. Anda perlu mengimplementasikan ulang logika preprocessing teks dan logika rekomendasi dalam expressjs di backend Anda.
