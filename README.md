# Automatic Image Clustering and Labeling using CLIP

## Overview

This project automatically organizes a dataset of images into meaningful
groups **without knowing the categories beforehand**. It uses the **CLIP
(Contrastive Language--Image Pretraining)** model to extract semantic
image features and then applies **KMeans clustering** to group visually
similar images.

After clustering, the system automatically **assigns labels to
clusters** by comparing cluster images with candidate text labels using
CLIP's text--image similarity.

This demonstrates **unsupervised dataset discovery**, where images can
be grouped and labeled even when dataset categories are unknown.

------------------------------------------------------------------------

## Features

-   Extracts semantic image embeddings using CLIP
-   Groups similar images using KMeans clustering
-   Automatically names clusters using text similarity
-   Works even if dataset categories are unknown
-   Compatible with Windows and Linux
-   Automatically creates folders for each discovered category

------------------------------------------------------------------------

## Project Structure

    clip_image_classifier/
    │
    ├── dataset/                # Input dataset (images)
    │
    ├── cluster_output/         # Output folders created after clustering
    │
    ├── cluster.py              # Main script
    │
    ├── requirements.txt        # Project dependencies
    │
    └── README.md

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone <repository_url>
cd clip_image_classifier
```

### 2. Create Virtual Environment (Recommended)

Linux / Mac:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Requirements

The project requires the following libraries:

-   torch
-   torchvision
-   numpy
-   pillow
-   scikit-learn
-   CLIP (installed from GitHub)

Example `requirements.txt`:

    torch
    torchvision
    numpy
    pillow
    scikit-learn
    git+https://github.com/openai/CLIP.git

------------------------------------------------------------------------

## How the System Works

### Step 1 --- Feature Extraction

Each image is passed through the CLIP model, which converts the image
into a **512-dimensional embedding vector**.

Example:

    dog.jpg → [0.32, 0.84, 0.11, ...]
    cat.jpg → [0.31, 0.82, 0.14, ...]
    chair.jpg → [0.72, 0.12, 0.44, ...]

Images with similar content produce similar embeddings.

------------------------------------------------------------------------

### Step 2 --- Feature Normalization

Embeddings are normalized to improve clustering performance.

    vector / ||vector||

------------------------------------------------------------------------

### Step 3 --- Clustering

The normalized embeddings are grouped using **KMeans clustering**.

Example clusters:

    Cluster 0 → dogs, cats
    Cluster 1 → humans
    Cluster 2 → furniture
    Cluster 3 → buildings
    Cluster 4 → landscapes

------------------------------------------------------------------------

### Step 4 --- Automatic Cluster Naming

To assign meaningful names to clusters, the system compares cluster
images with candidate text labels using CLIP.

Example candidate labels:

    dog
    cat
    rabbit
    bird
    human
    places

CLIP calculates similarity between images and labels, and the **highest
scoring label becomes the cluster name**.

Example:

    cluster_0 → dog
    cluster_1 → human
    cluster_2 → places

------------------------------------------------------------------------

## Running the Project

Place images inside the **dataset/** folder.

Then run:

``` bash
python cluster.py
```

------------------------------------------------------------------------

## Example Output

    cluster_output/
    │
    ├── dog/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │
    ├── human/
    │   ├── img3.jpg
    │
    ├── places/
    │   ├── img4.jpg

Images are automatically grouped and labeled.

------------------------------------------------------------------------

## Applications

This system can be used for:

-   Automatic dataset organization
-   Image search systems
-   Content moderation
-   Multimedia indexing
-   Unsupervised dataset discovery

