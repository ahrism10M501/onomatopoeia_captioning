import os
import numpy as np
import openl3
import librosa
import glob
import shutil
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU 확인
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# 모델 사전 로드
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="env", embedding_size=512
)

root_path = ''
embedding_dir = "embeddings"

sound_data = glob.glob(f"{root_path}\\train\\audio\\**\\*.wav", recursive=True) + \
             glob.glob(f"{root_path}\\valid\\audio\\**\\*.wav", recursive=True)
test_data = glob.glob(f'{root_path}\\test\\audio\\**\\*.wav')
print("총 파일 수:", len(test_data))

# 1. 임베딩 추출 및 저장
def extract_and_cache_openl3(file_path, model, save_dir="embeddings_test"):
    os.makedirs(save_dir, exist_ok=True)
    base = Path(file_path).stem
    save_path = Path(save_dir) / f"{base}.npy"

    if save_path.exists():
        return np.load(save_path)

    audio, sr = librosa.load(file_path, sr=None, mono=True)
    emb, _ = openl3.get_audio_embedding(audio, sr, input_repr="mel256", content_type="env",
                                        embedding_size=512, model=model)
    emb = np.mean(emb, axis=0)
    np.save(save_path, emb)
    return emb

# 2. 전체 파일에서 feature 로딩
def get_features_with_cache(file_list, model, embedding_dir="embeddings_test"):
    features = []
    filenames = []
    for file in file_list:
        if file.endswith(".wav"):
            feat = extract_and_cache_openl3(file, model, save_dir=embedding_dir)
            features.append(feat)
            filenames.append(file)
    return np.array(features), filenames

# 3. Elbow + Silhouette
def find_optimal_clusters(features, max_k=15):
    distortions = []
    silhouettes = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        preds = kmeans.fit_predict(features)
        distortions.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(features, preds))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouettes, marker='x', color='green')
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    return np.argmax(silhouettes) + 2  # index 보정

# 4. 클러스터 라벨별로 파일 분류
def save_clustered_files(filenames, labels, output_dir="clustering_test"):
    for file, label in zip(filenames, labels):
        cluster_folder = Path(output_dir) / f"cluster_{label}"
        cluster_folder.mkdir(parents=True, exist_ok=True)
        dest_path = cluster_folder / Path(file).name
        shutil.copy2(file, dest_path)
    print(f"[완료] 클러스터링 결과가 '{output_dir}'에 저장됨.")

# 5. 전체 실행 함수
def run_kmeans_clustering(file_list, model, output_dir="clustering"):
    features, filenames = get_features_with_cache(file_list, model)
    optimal_k = find_optimal_clusters(features)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(features)

    save_clustered_files(filenames, labels, output_dir)

    for fname, label in zip(filenames, labels):
        print(f"{fname} -> Cluster {label}")

    return labels

# 실행
run_kmeans_clustering(test_data, model)
