import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.decomposition import PCA
import mnist_reader
import numpy as np
import gzip

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ==================== DATA LOADING ====================
def load_mnist_images(path):
    """
    Load MNIST / Fashion-MNIST images from .gz ubyte file
    Returns: (n_samples, 784)
    """
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051")

        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)

    return data

def load_mnist_labels(path):
    """
    Load MNIST / Fashion-MNIST labels from .gz ubyte file
    Returns: (n_samples,)
    """
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049")

        num_labels = int.from_bytes(f.read(4), 'big')
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels

# Load dataset
print("=" * 60)
print("LOADING FASHION-MNIST DATASET")
print("=" * 60)
my_cur_path     = os.path.dirname(os.path.abspath(__file__))
data_path       = os.path.join(my_cur_path, 'dataset')
X_train_full    = load_mnist_images(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
y_train_full    = load_mnist_labels(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))
X_test_full     = load_mnist_images(os.path.join(data_path, 't10k-images-idx3-ubyte.gz'))
y_test_full     = load_mnist_labels(os.path.join(data_path, 't10k-labels-idx1-ubyte.gz'))

print("Training data shape:", X_train_full.shape)
print("Training labels shape:", y_train_full.shape)
print("Test data shape:", X_test_full.shape)
print("Test labels shape:", y_test_full.shape)

# Normalize
X_train_full = X_train_full / 255.0
X_test_full = X_test_full / 255.0

# Combine and filter to 5 classes
X = np.vstack((X_train_full, X_test_full))
y = np.hstack((y_train_full, y_test_full))
mask = y < 5
X, y = X[mask], y[mask]

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=5000, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=6000, random_state=42, stratify=y_train_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat']

print(f"\nDataset Split:")
print(f"  Training:   {X_train.shape[0]:,} samples")
print(f"  Validation: {X_val.shape[0]:,} samples")
print(f"  Test:       {X_test.shape[0]:,} samples")
print(f"  Classes:    {len(class_names)} ({', '.join(class_names)})")

# ==================== DIMENSIONALITY REDUCTION ====================
print("\n" + "=" * 60)
print("APPLYING PCA DIMENSIONALITY REDUCTION")
print("=" * 60)

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"\nOriginal dimensions: {X_train.shape[1]}")
print(f"Reduced dimensions:  {X_train_pca.shape[1]}")
print(f"Variance explained:  {pca.explained_variance_ratio_.sum():.2%}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')

# Only loop over the actual number of classes (5), not all 10 subplot slots
for i in range(len(class_names)):
    ax = axes.flat[i]
    # Find indices of samples belonging to class i
    class_indices = np.where(y_train == i)[0]
    if len(class_indices) == 0:
        # Skip if, for some reason, this class didn't appear in the training split
        continue
    idx = class_indices[0]
    ax.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    ax.set_title(class_names[i])
    ax.axis('off')
plt.tight_layout()
plt.show()

# ==================== LOGISTIC REGRESSION ====================
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

# Hyperparameter tuning
print("\nPerforming Grid Search for optimal hyperparameters...")
param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'saga']}
logreg = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_pca, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# Train final model on train + validation
best_logreg = grid.best_estimator_
X_combined = np.vstack((X_train_pca, X_val_pca))
y_combined = np.hstack((y_train, y_val))
best_logreg.fit(X_combined, y_combined)

# Predictions
y_pred_log = best_logreg.predict(X_test_pca)
y_proba_log = best_logreg.predict_proba(X_test_pca)
acc_log = accuracy_score(y_test, y_pred_log)

print(f"\n{'Test Accuracy:':<20} {acc_log:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=class_names))

# Loss curve (training progression)
print("\nGenerating loss curve...")
train_losses, val_losses = [], []
train_accs, val_accs = [], []

logreg_curve = LogisticRegression(
    multi_class='multinomial', solver='saga',
    C=grid.best_params_['C'], max_iter=1,
    warm_start=True, random_state=42
)

for epoch in range(1, 51):
    logreg_curve.max_iter = epoch
    logreg_curve.fit(X_train_pca, y_train)
    
    train_acc = logreg_curve.score(X_train_pca, y_train)
    val_acc = logreg_curve.score(X_val_pca, y_val)
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(1 - train_acc)
    val_losses.append(1 - val_acc)

# Visualizations for Logistic Regression
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Logistic Regression Analysis', fontsize=16, fontweight='bold')

# Loss curve
axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Loss Curve (1 - Accuracy)', fontweight='bold')
axes[0, 0].set_xlabel('Iterations')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curve
axes[0, 1].plot(train_accs, label='Training Accuracy', linewidth=2)
axes[0, 1].plot(val_accs, label='Validation Accuracy', linewidth=2)
axes[0, 1].axhline(y=acc_log, color='r', linestyle='--', label=f'Test Acc: {acc_log:.3f}')
axes[0, 1].set_title('Accuracy Curve', fontweight='bold')
axes[0, 1].set_xlabel('Iterations')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
im = axes[1, 0].imshow(cm_log, cmap='Blues', aspect='auto')
axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xticks(range(5))
axes[1, 0].set_yticks(range(5))
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].set_yticklabels(class_names)
for i in range(5):
    for j in range(5):
        axes[1, 0].text(j, i, str(cm_log[i, j]), ha='center', va='center',
                       color='white' if cm_log[i, j] > cm_log.max()/2 else 'black')
plt.colorbar(im, ax=axes[1, 0])

# ROC Curve
y_test_bin = pd.get_dummies(y_test)
for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin.iloc[:, i], y_proba_log[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})', linewidth=2)

axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1, 1].set_title('Multi-class ROC Curve', fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== K-MEANS CLUSTERING ====================
print("\n" + "=" * 60)
print("K-MEANS CLUSTERING MODEL")
print("=" * 60)

# Fit K-Means
print("\nFitting K-Means with 5 clusters...")
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_train_pca)

# Map clusters to labels
cluster_map = {}
for cluster in range(5):
    mask_cluster = (kmeans.labels_ == cluster)
    if mask_cluster.sum() > 0:
        cluster_map[cluster] = np.bincount(y_train[mask_cluster]).argmax()

print(f"\nCluster to Label Mapping: {cluster_map}")

# Predictions
cluster_pred = kmeans.predict(X_test_pca)
y_pred_kmeans = np.array([cluster_map.get(c, 0) for c in cluster_pred])
acc_kmeans = accuracy_score(y_test, y_pred_kmeans)

print(f"\n{'Test Accuracy:':<20} {acc_kmeans:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_kmeans, target_names=class_names))

# Inertia progression
print("\nGenerating inertia curve...")
inertias = []
for i in range(1, 101):
    km = KMeans(n_clusters=5, max_iter=i, n_init=1, random_state=42)
    km.fit(X_train_pca)
    inertias.append(km.inertia_)

# Distance-based probabilities for ROC
dists = kmeans.transform(X_test_pca)
scores_km = np.exp(-dists / dists.mean())
scores_km = scores_km / scores_km.sum(axis=1, keepdims=True)

# Visualizations for K-Means
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('K-Means Clustering Analysis', fontsize=16, fontweight='bold')

# Inertia curve
axes[0, 0].plot(range(1, 101), inertias, linewidth=2, color='orangered')
axes[0, 0].set_title('Inertia Curve', fontweight='bold')
axes[0, 0].set_xlabel('Iterations')
axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0, 0].grid(True, alpha=0.3)

# Cluster sizes
cluster_sizes = [np.sum(kmeans.labels_ == i) for i in range(5)]
axes[0, 1].bar(range(5), cluster_sizes, color='coral', edgecolor='black')
axes[0, 1].set_title('Cluster Size Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Number of Samples')
axes[0, 1].set_xticks(range(5))
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cluster_sizes):
    axes[0, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Confusion Matrix
cm_km = confusion_matrix(y_test, y_pred_kmeans)
im = axes[1, 0].imshow(cm_km, cmap='Reds', aspect='auto')
axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xticks(range(5))
axes[1, 0].set_yticks(range(5))
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].set_yticklabels(class_names)
for i in range(5):
    for j in range(5):
        axes[1, 0].text(j, i, str(cm_km[i, j]), ha='center', va='center',
                       color='white' if cm_km[i, j] > cm_km.max()/2 else 'black')
plt.colorbar(im, ax=axes[1, 0])

# ROC Curve
for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin.iloc[:, i], scores_km[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})', linewidth=2)

axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1, 1].set_title('Multi-class ROC Curve', fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== MODEL COMPARISON ====================
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_data = {
    'Model': ['Logistic Regression', 'K-Means'],
    'Test Accuracy': [acc_log, acc_kmeans],
    'Type': ['Supervised', 'Unsupervised']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Accuracy comparison
models = ['Logistic\nRegression', 'K-Means']
accuracies = [acc_log, acc_kmeans]
colors = ['steelblue', 'coral']

bars = axes[0].bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
axes[0].set_title('Test Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Per-class accuracy
class_acc_log = []
class_acc_km = []
for i in range(5):
    mask = y_test == i
    class_acc_log.append(accuracy_score(y_test[mask], y_pred_log[mask]))
    class_acc_km.append(accuracy_score(y_test[mask], y_pred_kmeans[mask]))

x = np.arange(len(class_names))
width = 0.35
axes[1].bar(x - width/2, class_acc_log, width, label='Logistic Reg', color='steelblue', edgecolor='black')
axes[1].bar(x + width/2, class_acc_km, width, label='K-Means', color='coral', edgecolor='black')
axes[1].set_title('Per-Class Accuracy', fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Class')
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)