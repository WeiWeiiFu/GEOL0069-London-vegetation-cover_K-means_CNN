# GEOL0069-London-vegetation-cover_K-means_CNN
Using k-means and CNN classification to analyse SENTINEL-2 satellite data for London vegetation cover.


![Unknown-5](https://github.com/user-attachments/assets/c8de5ad7-1753-4d84-b9a3-1c9be6d16de9)

## A description of the problem to be tackled 
Urban vegetation is crucial for environmental quality and city planning, yet mapping green spaces at scale is challenging. This project tackles automatic vegetation cover classification in London using Sentinel-2 satellite imagery. We exploit the Normalized Difference Vegetation Index (NDVI) – a spectral index highlighting live green vegetation – to differentiate vegetated and non-vegetated land cover. Our goal is to compare an unsupervised approach (K-means clustering on NDVI data) with a weakly-supervised deep learning approach (U-Net convolutional neural network) for detecting vegetation. By leveraging Earth observation data and AI, we demonstrate how urban green areas can be mapped with minimal manual labels, which is valuable for urban planners and ecologists.
Urban vegetation plays a vital role in mitigating the adverse effects of urbanization and climate change. It significantly contributes to improving thermal comfort, carbon storage capacity, rainwater infiltration, pollutant absorption and biodiversity enhancement in cities. With the acceleration of urbanization, it is expected that by 2050, 68% of the world's population will live in urban areas, which exacerbates challenges such as urban heat islands and makes urban green space a priority that needs to be addressed. The goal of this project is to use artificial intelligence technology to map and quantify the vegetation cover in Greater London from Sentinel-2 imagery obtained from Google EE.

## Data sources
Google EE
https://console.cloud.google.com/earth-engine/configuration;success=true?inv=1&invt=AbzGzg&project=silver-bird-461614-i1
## Methodology
## NDVI - Normalized Difference Vegetation Index
The Normalized Difference Vegetation Index (NDVI) is a widely used metric for assessing the health and density of vegetation. It is calculated using data from the red and near-infrared bands of a remote sensing sensor, such as a satellite. NDVI values range from -1 to 1, with higher positive values indicating denser, healthier vegetation.
Calculation: NDVI is calculated using the following formula: (NIR - RED) / (NIR + RED), where NIR is the reflectance in the near-infrared band and RED is the reflectance in the red band.
Interpretation: Positive NDVI values (0 to 1) generally indicate vegetation, with higher values indicating more active and healthy vegetation. Values close to 0 (or slightly negative) often represent barren areas, soil, or water. Negative values (close to -1) are typically associated with water bodies.
<img width="461" alt="Screenshot 2025-06-03 at 08 18 48" src="https://github.com/user-attachments/assets/795cae92-4f6d-4e1c-994f-d0de933b11ff" />
## K-Means clustering
### Key Components of K-means
Choosing K: The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.

Centroids Initialization: The initial placement of the centroids can affect the final results.

Assignment Step: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.

Update Step: The centroids are recomputed as the center of all the data points assigned to the respective cluster.

K-means clustering is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

Advantages of K-means:
Efficiency: K-means is computationally efficient. Ease of interpretation: The results of k-means clustering are easy to understand and interpret.
import rasterio, numpy as np
from sklearn.cluster import KMeans

ndvi_path = '/content/drive/MyDrive/London_NDVI.tif'
with rasterio.open(ndvi_path) as src:
    ndvi = src.read(1).astype('float32')


if ndvi.max() > 1.2 or ndvi.min() < -1.2:
    ndvi /= 10000.0
ndvi = np.clip(ndvi, -1, 1)


ndvi[roi_mask == 0] = np.nan

 3) K-means
X = ndvi.reshape(-1, 1)
kmeans = KMeans(
    n_clusters=3,
    n_init=20,
    random_state=42
).fit(X)

labels = kmeans.labels_.reshape(ndvi.shape)
cluster_means = [ndvi[labels == i].mean() for i in range(3)]
veg_cluster = int(np.argmax(cluster_means))

veg_mask  = (labels == veg_cluster)
coverage  = np.nanmean(veg_mask) * 100
print(f" London vegetation cover：{coverage:.2f}%")

import matplotlib.pyplot as plt
import numpy as np


 ndvi
 labels
 veg_mask
 kmeans

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

 ① NDVI
im0 = ax[0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
ax[0].set_title('NDVI', fontsize=12)
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

 ② K-means labels
im1 = ax[1].imshow(labels, cmap='tab10')
ax[1].set_title(f'K-means labels (k={kmeans.n_clusters})', fontsize=12)
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

 ③ K- means London vegetation cover
im2 = ax[2].imshow(veg_mask, cmap='Greens')
ax[2].set_title('K-means', fontsize=12)
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

<img width="1215" alt="Screenshot 2025-06-03 at 08 22 07" src="https://github.com/user-attachments/assets/92fb00b9-9da1-4794-9b3b-31af0502851b" />

## CNN - Convolutional Neural Networksv
### Key Components of CNN

Convolutional Layer : This is the core building block of a CNN. It slides a filter (smaller in size than the input data) over the input data (like an image) to produce a feature map or convolved feature. The primary purpose of a convolution is to extract features from the input data.

Pooling Layer: Pooling layers are used to reduce the dimensions of the feature maps, thereby reducing the number of parameters and computation in the network. The most common type of pooling is max pooling.

Fully Connected Layer: After several convolutional and pooling layers, the final classification is done using one or more fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular neural networks.

Activation Functions: Non-linearity is introduced into the CNN using activation functions. The Rectified Linear Unit (ReLU) is the most commonly used activation function in CNNs.

Convolutional Neural Networks, commonly known as CNNs, are a class of deep neural networks specially designed to process data with grid-like topology, such as images. Originating from the visual cortex’s biological processes, CNNs are revolutionising the way we understand and interpret visual data.

Advantages of CNNs:
Parameter Sharing: A feature detector (filter) that’s useful in one part of the image can be useful in another part of the image Sparsity of Connections: In each layer, each output value depends only on a small number of input values, making the computation more efficient.

PATCHES        = 30_000
WINDOW_SIZE    = 64
BATCH_SIZE     = 32
EPOCHS_MAX     = 3
THRESH_VEG     = 0.60
SEED           = 42

1. Imports & reproducibility
import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(SEED)
tf.random.set_seed(SEED)

2. Load & normalise NDVI raster
with rasterio.open(NDVI_PATH) as src:
    ndvi      = src.read(1).astype(np.float32)
    ndvi_meta = src.profile

if ndvi.max() > 1.2 or ndvi.min() < -1.2:
    ndvi /= 10_000.0
ndvi = np.clip(ndvi, -1.0, 1.0)
ndvi_norm = (ndvi + 1.0) / 2.0

3. Weak labels

veg_mask = (ndvi > THRESH_VEG).astype(np.float32)


4. Patch sampler


def extract_patches(img, msk, win=64, n=30_000):
    h, w = img.shape; pad = win//2
    img_p = np.pad(img, pad, mode="reflect"); msk_p = np.pad(msk, pad, mode="reflect")
    X = np.empty((n, win, win, 1), np.float32)
    y = np.empty_like(X)
    for k in range(n):
        i = np.random.randint(pad, h+pad); j = np.random.randint(pad, w+pad)
        X[k,...,0] = img_p[i-pad:i+pad, j-pad:j+pad]
        y[k,...,0] = msk_p[i-pad:i+pad, j-pad:j+pad]
    return X, y

X, y = extract_patches(ndvi_norm, veg_mask, WINDOW_SIZE, PATCHES)
strat = (y.mean(axis=(1,2,3)) > 0.10)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, stratify=strat, random_state=SEED)


5. Tiny U‑Net (dynamic) – clear old graph first


wipe previous graphs (prevents shape lock-in during iterative editing)
tf.keras.backend.clear_session()

def unet_small():
    """Fully‑convolutional U‑Net that ingests any H×W×1; depths=2."""
    inp = layers.Input(shape=(None, None, 1))
    # Down 1
    c1 = layers.Conv2D(32,3,padding='same',activation='relu')(inp)
    p1 = layers.MaxPool2D()(c1)
    # Bottleneck
    c2 = layers.Conv2D(64,3,padding='same',activation='relu')(p1)
    # Up 1
    u1  = layers.UpSampling2D()(c2)
    m1  = layers.Concatenate()([u1, c1])
    c3  = layers.Conv2D(32,3,padding='same',activation='relu')(m1)
    out = layers.Conv2D(1,1,activation='sigmoid')(c3)
    return models.Model(inp, out, name="unet_small_dyn")

model = unet_small()


6. Loss & compile (broadcast‑safe)


def dice_loss(y_true, y_pred, smooth=1.):
    y_true = tf.cast(y_true, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union  = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    return 1. - (2.*inter + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # BCE per‑pixel ➜ reduce to (batch,)
    bce_map = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce     = tf.reduce_mean(bce_map, axis=[1,2,3])
    dice    = dice_loss(y_true, y_pred)
    return 0.6*bce + 0.4*dice

model.compile(optimizer='adam', loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

7. Training

callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(patience=3)]

history = model.fit(X_tr, y_tr,
                    validation_data=(X_va, y_va),
                    epochs=EPOCHS_MAX,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=1)

8. Full‑image inference & coverage

prob = model.predict(ndvi_norm[None,...,None], batch_size=1)[0,...,0]
mask = (prob > 0.5).astype(np.uint8)
print(f"\n Vegetation cover: {mask.mean()*100:.2f}%")


9. Save GeoTIFF

meta = ndvi_meta.copy(); meta.update(dtype=rasterio.uint8, count=1)
with rasterio.open("London_veg_mask_cnn.tif", "w", **meta) as dst:
    dst.write(mask, 1)

10. print

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,3,figsize=(16,5))
        ax[0].imshow(ndvi, cmap='RdYlGn', vmin=-1,vmax=1); ax[0].set_title('NDVI'); ax[0].axis('off')
        ax[1].imshow(prob, cmap='viridis', vmin=0,vmax=1); ax[1].set_title('Vegetation probability'); ax[1].axis('off')
        ax[2].imshow(mask, cmap='Greens'); ax[2].set_title('CNN'); ax[2].axis('off')
        plt.tight_layout(); plt.show()
    
<img width="1240" alt="Screenshot 2025-06-03 at 08 26 02" src="https://github.com/user-attachments/assets/1b22ce8f-767e-42f3-bcf4-adefdc68d55b" />

### London Vegetation Cover 

| Method              | Coverage (%) |
|---------------------|-------------:|
| K-means (k = 3)     |        43.88 |
| CNN                 |        40.10 |


## Getting Start
This project was created using Google Colab. To get a copy of this project up and running, follow these steps:

1.Clone or download the repository

2.Open the notebook in Google Colab

3.Acquire the Sentinel-2 NDVI GeoTIFF file

4.Ensure the notebook paths point to your NDVI file

5.Adjust any other file paths or parameters

6.Install required Python packages 

7.Run the notebook cells in order

8.Replace with your own imagery
	•	If you want to experiment on a different city or region, simply supply your own NDVI GeoTIFF (with the same coordinate system and dimensions) and update NDVI_PATH.
	•	The clustering and U-Net workflow remain unchanged, but you may need to adjust the patch extraction size or threshold (THRESH_VEG) based on local NDVI distributions.
## A figure illustrating the remote sensing technique
<img width="612" alt="Screenshot 2025-06-03 at 08 51 12" src="https://github.com/user-attachments/assets/52050438-0102-4e68-bda3-160fe476210f" />

## A figure illustrating the AI algorithm and its implementation
<img width="599" alt="Screenshot 2025-06-03 at 08 54 11" src="https://github.com/user-attachments/assets/079d8b93-af56-45ae-991c-c024d1dd9336" />

## Environmental Cost Assessment
This section quantifies the energy consumption and associated CO₂ emissions incurred by our London vegetation‐cover project (Sentinel-2 NDVI, K-means, and a lightweight U-Net CNN). We focus on the main sources of energy use—unsupervised K-means clustering (CPU), U-Net model training (GPU), and full-image inference (GPU/CPU)—and convert each to an estimated carbon footprint.
1. K-means Clustering (CPU)
	•	Data size: One Sentinel-2 NDVI raster for Greater London (≈ 452 × 940 pixels ≈ 425 000 values).
	•	Hardware: Google Colab’s CPU (≈ 45 W average power draw).
	•	Measured runtime: ~ 30 seconds.
	•	Energy consumed:45 W × (30 s ÷ 3600 s/h) = 0.000375 kWh
        •	CO₂ emissions (0.4 kg CO₂/kWh): 0.000375 kWh × 0.4 kg CO₂/kWh = 0.00015 kg CO₂ ≈ 0.15 g CO₂
2. CNN Training (GPU)
	•	Patch sampling: 30 000 patches of size 64 × 64.
	•	Model: A small U-Net (approx. two down-sampling and two up-sampling layers).
	•	Hardware: Google Colab Tesla T4 GPU (≈ 70 W average when training).
	•	Configuration:
	  •	Batch size: 32
	  •	Epochs: 10 (observed ~ 15 min per epoch; total ~ 2.5 h)
	•	Energy used: 70 W × 2.5 h = 0.175 kWh
	•	CO₂ emissions (0.4 kg CO₂/kWh): 0.175 kWh × 0.4 kg CO₂/kWh = 0.07 kg CO₂ = 70 g CO₂
3. Full-Image Inference (GPU/CPU)
	•	Task: Run model.predict() on the entire 452 × 940 NDVI image.
	•	Hardware: Tesla T4 GPU (or fallback to CPU).
	•	Runtime: ~ 30 seconds.
	•	Energy used: 70 W × (30 s ÷ 3600 s/h) = 0.000583 kWh
	•	CO₂ emissions (0.4 kg CO₂/kWh):0.000583 kWh × 0.4 kg CO₂/kWh = 0.000233 kg CO₂ ≈ 0.23 g CO₂
### 4. Total Energy & CO₂ Emissions

|                            | Energy (kWh)  | CO₂ Emissions (g) |
|----------------------------|--------------:|------------------:|
| K-means (CPU)              |       0.00038 |             0.15  |
| CNN Training (GPU)         |       0.17500 |            70.00  |
| Full-Image Inference (GPU) |       0.00058 |             0.23  |
| **Total**                  |     **0.1760**|          **70.38**|

- **Total energy**: 0.176 kWh  
- **Total CO₂**: 0.0738 kg (≈ 70 g)
Context: 70 g CO₂ is roughly the same as charging a smartphone several times or driving an electric car for a few hundred meters, so this workflow is very lightweight in environmental impact.



## Video
https://youtu.be/hDvfrgatk9Y?feature=shared

## LIcense
The Unlicense
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. In general, Sentinel data is free and open to the public under EU law. Please consider the Copernicus Sentinel Data Terms and Conditions when using Copernicus Sentinel data.
## Contact
WEI FU - zcfbwfu@ucl.ac.uk/wweiweifu@163.com
Link - https://github.com/WeiWeiiFu/GEOL0069-London-vegetation-cover_K-means_CNN.git

## Acknowledgments
This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados and Weibin Chen.
## References:
Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. http://www.deeplearningbook.org.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–444, May 2015. doi:10.1038/nature14539.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 2012.

James MacQueen and others. Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth 

Berkeley symposium on mathematical statistics and probability, volume 1, 281–297. Oakland, CA, USA, 1967.





