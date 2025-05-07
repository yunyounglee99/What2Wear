# What2Wear

## Example Results

<img width="414" alt="image" src="https://github.com/user-attachments/assets/d473a5f4-57ea-418c-8a7c-cf8293b860b2">

<img width="363" alt="image" src="https://github.com/user-attachments/assets/205b7957-676e-41f3-94b3-b8dde1e3455f">

## Recomendation Algorithm

<img width="599" alt="스크린샷 2025-05-07 오후 1 03 42" src="https://github.com/user-attachments/assets/a32d58ee-c5bd-492c-a6b9-ded365107c3d" />


# 1. Introduction

---
This project is an artificial intelligence model development project that recognizes the details of individual fashion items and understands outfits that can be worn together through compatibility learning, rather than just single items. This project proposes a model that learns compatibility and a new form of recommendation system utilizing it.

# 2. Related Work & Background

---

### FASHION CLIP

<img width="909" alt="image" src="https://github.com/user-attachments/assets/7b827354-2a6c-4e10-bf7d-c51bdab2623f">


- FASHION CLIP is a CLIP-based model trained on a dataset of single fashion image items and caption pairs.

<img width="877" alt="image 1" src="https://github.com/user-attachments/assets/b88b405b-f8bd-44d8-ac84-91352db9606b">


- Through heatmaps, we can see that FASHION CLIP recognizes the details of fashion items.

# 3. Method

---

### Generate outfit(modified) vector

<img width="910" alt="image 2" src="https://github.com/user-attachments/assets/e77bdda7-71cc-4377-88db-8b28f9d1e419">


- For training, we prepared full-body outfit images with white backgrounds (processed by SAM) that include tops, bottoms, and shoes.
- The outfit images were cropped at appropriate ratios into tops, bottoms, and shoes so that the image encoder could better understand the single images within the outfit.

<img width="686" alt="image 3" src="https://github.com/user-attachments/assets/32db773f-5f94-4921-89df-e010a7a1d365">


1. The cropped outfit images (tops, bottoms, shoes) were each encoded into 512 dimensions using the Image Encoder (FASHION CLIP).
2. We concatenated the encoded vectors for each category to create an outfit vector with dimensions (3, 512)
3. We created a modified vector by randomly selecting two categories from the outfit vector (e.g., [top, bottom] or [top, shoes], etc.) and replacing them with items drawn from random data.
4. The outfit vector is used as the 'correct answer vector' to learn clear compatibility, while the modified vector is used to learn degrees of compatibility.

### Training

<img width="707" alt="image 4" src="https://github.com/user-attachments/assets/ffc65012-6aaf-4d3a-b2a9-f4c99715f5f6">


- Training learns compatibility in a single latent space.
    - Both the outfit vector and modified vector learn the same compatibility in the same latent space.
    - During training, contrastive loss is performed for each category combination. We perform loss calculation 3 times for (top, bottom), (bottom, shoes), and (top, shoes), then calculate the average
    - The outfit vector is used to learn clear compatibility, with loss minimized when each similarity is maximized.
    - The modified vector is used to learn moderate compatibility, designed to minimize loss when above a certain threshold (indicating compatibility) and maximize loss when below a certain threshold (indicating incompatibility).
    - The final loss is the sum of losses from both vectors.
- We used dropout and layernorm to prevent overfitting.

# 4. Experiments

---

<img width="894" alt="image 6" src="https://github.com/user-attachments/assets/c2e43cd8-c399-4b85-a9ca-f138e86fa043">


- We can see that the outfit vector similarity is distributed more widely than the modified vector (which appears to follow a Gaussian distribution).
- Through the similarity between outfit vectors and modified vectors, we can confirm that clearly compatible items are treated as compatible, and incompatible items are treated as incompatible.

<img width="610" alt="image 7" src="https://github.com/user-attachments/assets/d1fd2282-352a-403c-bc93-2f341cb43ad9">

-  However, we found that performance was good only at low thresholds and deteriorated as the threshold increased.



# 5. Discussion

---

1. Outfit data often includes fashion items that are partially obscured rather than fully visible. To ensure FASHION CLIP recognizes these clearly, fine-tuning FASHION CLIP with data augmentation appears necessary.
2. Since we trained the outfit vector and modified vector simultaneously, there's a high possibility that the modified vector learned degrees of compatibility without fully learning compatibility itself. We are considering training the outfit vector first, then training the modified vector.
3. Since we trained the outfit vector and modified vector in the same latent space, there's a high possibility of learning unclear compatibility. We are considering using different embedding spaces for each category or exploring other methods.
4. The quality and quantity of outfit data (Musinsa crawling → processed by SAM / total of 23,000 items) may have been problematic. We are looking for solutions to address this.
