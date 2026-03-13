import os
import shutil
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

dataset_path = r"C:\Users\hp\Desktop\clip_image_classifier\dataset"
output_path = r"C:\Users\hp\Desktop\clip_image_classifier\cluster_output"

num_clusters = 6

image_paths = []
image_features = []

print("Extracting CLIP features...")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        path = os.path.join(root, file)

        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(image)

            features = features / features.norm(dim=-1, keepdim=True)

            image_paths.append(path)
            image_features.append(features.cpu().numpy()[0])

        except:
            print("Skipping:", file)

image_features = np.array(image_features)

print("Running clustering...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42)

labels = kmeans.fit_predict(image_features)

# Create cluster folders
cluster_folders = []

for i in range(num_clusters):

    folder = os.path.join(output_path, f"cluster_{i}")

    os.makedirs(folder, exist_ok=True)

    cluster_folders.append(folder)

# Move images into clusters
cluster_images = {i: [] for i in range(num_clusters)}

for i, path in enumerate(image_paths):

    cluster_id = labels[i]

    filename = os.path.basename(path)

    destination = os.path.join(cluster_folders[cluster_id], filename)

    shutil.copy(path, destination)

    cluster_images[cluster_id].append(path)

print("Clusters created")

# -----------------------------
# CLUSTER NAMING
# -----------------------------

print("Naming clusters...")

candidate_labels = [
    "dog","cat","rabbit","bird",
    "human","places"
]

text_tokens = clip.tokenize(candidate_labels).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)

for cluster_id in range(num_clusters):

    images = cluster_images[cluster_id][:5]  # sample few images

    similarities = []

    for path in images:

        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(image)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

        similarity = (image_feature @ text_features.T).softmax(dim=-1)

        similarities.append(similarity.cpu().numpy()[0])

    avg_similarity = np.mean(similarities, axis=0)

    best_label = candidate_labels[np.argmax(avg_similarity)]

    old_folder = cluster_folders[cluster_id]

    new_folder = os.path.join(output_path, best_label)

    if not os.path.exists(new_folder):
        os.rename(old_folder, new_folder)
    else:
        new_folder = new_folder + f"_{cluster_id}"
        os.rename(old_folder, new_folder)

print("Clusters named successfully!")