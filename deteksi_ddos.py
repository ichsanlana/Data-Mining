import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Fungsi untuk membaca dataset
def read_dataset(file_path):
    dataset = pd.read_csv('d:/Semester 6/Data Mining/Tugas akhir/archive_2/ddos_imbalanced/unbalaced_20_80_dataset.csv')
    return dataset

# Fungsi untuk memilih fitur yang akan digunakan
def select_features(dataset):
    selected_features = dataset[['Fwd Pkt Len Mean', 'Tot Fwd Pkts', 'Fwd IAT Std', 'Fwd Pkt Len Std', 'Flow Pkts/s', 'Flow Duration', 'Label']]
    return selected_features

# Fungsi untuk membagi dataset menjadi fitur dan label
def split_features_labels(dataset):
    features = dataset.drop(columns=['Label'])
    labels = dataset['Label']
    return features, labels

# Fungsi untuk melakukan normalisasi pada fitur menggunakan Min-Max Scaler
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Fungsi untuk melatih model menggunakan MLP Regressor dengan algoritma lbfgs
def train_mlp_regressor(X_train, y_train):
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(4, 4, 4), solver='lbfgs')
    mlp_regressor.fit(X_train, y_train)
    return mlp_regressor

# Fungsi untuk melakukan prediksi menggunakan model yang telah dilatih
def predict_labels(model, X_test):
    predicted_labels = model.predict(X_test)
    return predicted_labels

# Fungsi untuk menghitung mean squared error (MSE) sebagai ukuran keberhasilan algoritma
def calculate_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

# Main function
if __name__ == "__main__":
    # Baca dataset
    dataset = read_dataset("d:/Semester 6/Data Mining/Tugas akhir/archive_2/ddos_imbalanced/unbalaced_20_80_dataset.csv")

    # Pilih fitur yang akan digunakan
    selected_features = select_features(dataset)

    # Bagi dataset menjadi fitur dan label
    features, labels = split_features_labels(selected_features)

    # Lakukan normalisasi pada fitur
    normalized_features = normalize_features(features)

    # Bagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)

    # Latih model dengan algoritma lbfgs dan arsitektur Hidden Layer 3 lapisan dengan 4 neuron di setiap lapisan
    model = train_mlp_regressor(X_train, y_train)

    # Lakukan prediksi pada data uji
    predicted_labels = predict_labels(model, X_test)

    # Hitung MSE sebagai ukuran keberhasilan algoritma
    mse = calculate_mse(y_test, predicted_labels)
    print("Mean Squared Error:", mse)
