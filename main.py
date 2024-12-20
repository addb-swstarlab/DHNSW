import os
import numpy as np
import pandas as pd
import time
import tracemalloc
from dhnsw import HNSW, DynamicHNSW
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

# Initialize key parameters
M_VANILLA = 16  # Number of connections per node for vanilla HNSW
EF_VANILLA_BASE = 100  # Default candidate list size for vanilla HNSW
SCALE_FACTOR = 1.5  # Scaling factor for parameter adjustments
EF_SCALE_FACTOR = 100
K = M_VANILLA  # Number of neighbors for density estimation

NUM_VECTORS = 1000  # Number of training vectors
NUM_QUERY_VECTORS = 100  # Number of query vectors
TOP_K = 100  # Top-K neighbors to retrieve
GLOVE_DIM = 300  # Dimensionality of GloVe dataset

DATASET_DIR = "./"  # Root directory for all datasets

# Dataset execution flags
RUN_DATASETS = {
    "mnist": True,
    "glove100k": True,
    "sift1m": True,
    "gist": True,
}

# Dataset loading functions
def load_gist(data_dir=f"{DATASET_DIR}/gist", num_train_vectors=NUM_VECTORS, num_query_vectors=NUM_QUERY_VECTORS):
    base_file_path = os.path.join(data_dir, 'gist_base.fvecs')
    query_file_path = os.path.join(data_dir, 'gist_query.fvecs')

    def read_fvecs(file_path, num_vectors):
        with open(file_path, 'rb') as f:
            d = np.fromfile(f, dtype=np.int32, count=1)[0]  # First value is dimension
            vectors = []
            while len(vectors) < num_vectors:
                vec = np.fromfile(f, dtype=np.float32, count=d)
                if vec.size != d:
                    break
                vectors.append(vec)
            return np.array(vectors)

    train = read_fvecs(base_file_path, num_train_vectors)
    test = read_fvecs(query_file_path, num_query_vectors)
    return train, test

def load_mnist(data_dir=f"{DATASET_DIR}/mnist", num_train_vectors=NUM_VECTORS, num_query_vectors=NUM_QUERY_VECTORS):
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')

    def read_images(file_path, num_vectors):
        with open(file_path, 'rb') as f:
            _ = np.frombuffer(f.read(4), dtype=np.uint8)  # Magic number
            num = np.frombuffer(f.read(4), dtype='>i4')[0]  # Number of images
            rows = np.frombuffer(f.read(4), dtype='>i4')[0]  # Number of rows
            cols = np.frombuffer(f.read(4), dtype='>i4')[0]  # Number of columns
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images[:num_vectors]

    def read_labels(file_path, num_vectors):
        with open(file_path, 'rb') as f:
            _ = np.frombuffer(f.read(4), dtype=np.uint8)  # Magic number
            num = np.frombuffer(f.read(4), dtype='>i4')[0]  # Number of labels
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels[:num_vectors]

    train_images = read_images(train_images_path, num_train_vectors)
    test_images = read_images(test_images_path, num_query_vectors)

    return train_images, test_images

def load_sift1m(data_dir=f"{DATASET_DIR}/sift1m", num_train_vectors=NUM_VECTORS, num_query_vectors=NUM_QUERY_VECTORS):
    base_file_path = os.path.join(data_dir, 'sift_base.fvecs')
    query_file_path = os.path.join(data_dir, 'sift_query.fvecs')

    def read_fvecs(file_path, num_vectors):
        with open(file_path, 'rb') as f:
            d = np.fromfile(f, dtype=np.int32, count=1)[0]  # First value is dimension
            vectors = []
            while len(vectors) < num_vectors:
                vec = np.fromfile(f, dtype=np.float32, count=d)
                if vec.size != d:
                    break
                vectors.append(vec)
            return np.array(vectors)

    train = read_fvecs(base_file_path, num_train_vectors)
    test = read_fvecs(query_file_path, num_query_vectors)
    return train, test

def load_glove100k(data_dir=f"{DATASET_DIR}/glove100k", num_vectors=NUM_VECTORS, num_query_vectors=NUM_QUERY_VECTORS, dim=300):
    glove_file_path = f"{data_dir}/glove.6B.{dim}d.txt"

    def read_glove(file_path, num_vectors):
        vectors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_vectors:
                    break
                values = line.split()
                vectors.append([float(x) for x in values[1:]])
        return np.array(vectors)

    train_data = read_glove(glove_file_path, num_vectors)
    query_data = read_glove(glove_file_path, num_query_vectors)
    return train_data, query_data

# Normalize data
def normalize_data(data):
    """Normalize the dataset to range [0, 1]."""
    return data / 255.0

# Calculate density using Random Projection
def calculate_density_random_projection(data, k=K, n_components=None):
    """Calculate density using Random Projection and KNN distances."""
    if n_components is None:
        n_components = max(1, data.shape[1] // 3)
    rp = GaussianRandomProjection(n_components=n_components)
    reduced_data = rp.fit_transform(data)
    knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    knn.fit(reduced_data)
    distances, _ = knn.kneighbors(reduced_data)
    densities = np.mean(distances, axis=1)
    return densities

# Adjust EF value based on dataset dimensionality
def adjust_ef_by_dim(ef_vanilla_base, data_dim):
    """Adjust EF based on the dimensionality of the dataset."""
    adjustment_value = (data_dim // EF_SCALE_FACTOR) ** 2
    return ef_vanilla_base + adjustment_value

# Set dynamic HNSW parameters based on density statistics
def set_dynamic_hnsw_params_by_std(densities, m_vanilla, ef_vanilla, scale_factor=SCALE_FACTOR):
    """Set dynamic HNSW parameters based on density statistics."""
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    cv = std_density / mean_density  # Coefficient of Variation
    m_start = max(2, int(m_vanilla - m_vanilla * cv * scale_factor))
    m_end = int(m_vanilla + m_vanilla * cv * scale_factor)
    ef_start = max(10, int(ef_vanilla - ef_vanilla * cv * scale_factor))
    ef_end = int(ef_vanilla + ef_vanilla * cv * scale_factor)

    print(f"Coefficient of Variation (CV): {cv:.4f}, M range: [{m_start}, {m_end}], EF range: [{ef_start}, {ef_end}]")
    return m_start, m_end, ef_start, ef_end

# Recall calculation
def calculate_recall(true_neighbors, retrieved_neighbors):
    true_set = set(true_neighbors)
    retrieved_set = set(retrieved_neighbors)
    recall = len(true_set & retrieved_set) / len(true_set)
    return recall * 100  # Return as percentage

# Measure HNSW performance
def measure_performance(hnsw_class, data, queries, true_neighbors, description, k, densities=None, ef_vanilla=None):
    if ef_vanilla is None:
        ef_vanilla = EF_VANILLA_BASE
        
    """Measure the performance of an HNSW implementation."""
    tracemalloc.start()
    start_time = time.time()
    if hnsw_class == DynamicHNSW:
        m_start, m_end, ef_start, ef_end = set_dynamic_hnsw_params_by_std(densities, M_VANILLA, ef_vanilla)
        hnsw = hnsw_class('l2', densities, m_start=m_start, m_end=m_end, ef_start=ef_start, ef_end=ef_end)
    else:
        hnsw = hnsw_class('l2', m=M_VANILLA, ef=ef_vanilla)

    for point in data:
        hnsw.add(point)
    build_time = time.time() - start_time

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = peak / 10**6

    recalls = []
    for i, query in enumerate(queries):
        search_results = hnsw.search(query, k)
        search_indices = [idx for idx, _ in search_results]
        recall = calculate_recall(true_neighbors[i], search_indices)
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    print(f"{description} - Build Time: {build_time:.2f}s, Memory Usage: {memory_usage:.2f} MB, Recall: {avg_recall:.2f}%")
    return build_time, avg_recall, memory_usage

# Load datasets
data_loaders = {
    "mnist": load_mnist,
    "glove100k": load_glove100k,
    "sift1m": load_sift1m,
    "gist": load_gist
}

datasets = {}
for name, loader in data_loaders.items():
    if RUN_DATASETS[name]:
        print(f"Loading {name} dataset...")
        datasets[name] = loader()

# Run experiments
for name, (train_data, query_data) in datasets.items():
    print(f"\nRunning experiments for {name}...")
    ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train_data.shape[1])
    densities = calculate_density_random_projection(train_data, k=K)

    knn = NearestNeighbors(n_neighbors=TOP_K, algorithm='brute', metric='euclidean')
    knn.fit(train_data)
    true_neighbors = [knn.kneighbors([query], n_neighbors=TOP_K, return_distance=False)[0] for query in query_data]

    measure_performance(DynamicHNSW, train_data, query_data, true_neighbors, f"Dynamic HNSW ({name})", k=TOP_K, densities=densities, ef_vanilla=ef_vanilla)
    measure_performance(HNSW, train_data, query_data, true_neighbors, f"Vanilla HNSW ({name})", k=TOP_K)
