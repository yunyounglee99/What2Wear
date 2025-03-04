# What2Wear

# 1. Introduction

---
패션 도메인에서 단일 아이템의 디테일을 인식하고, 어울림 학습을 통해 하나의 단일 아이템이 아닌, 함께 입을 수 있는 코디를 이해하는 인공지능 모델 개발 프로젝트입니다. 이 프로젝트에서는 어울림을 학습하는 모델과 이를 활용한 새로운 형태의 추천 시스템을 제안합니다.

# 2. Related Work & Background

---

### FASHION CLIP

<img width="909" alt="image" src="https://github.com/user-attachments/assets/7b827354-2a6c-4e10-bf7d-c51bdab2623f">


- FASHION CLIP은 단일 패션 이미지 아이템, 캡션 쌍 데이터셋으로 학습된 CLIP 기반 모델입니다.

<img width="877" alt="image 1" src="https://github.com/user-attachments/assets/b88b405b-f8bd-44d8-ac84-91352db9606b">


- heatmap을 통해 FASHION CLIP이 패션아이템의 디테일을 인식하고 있는 것을 알 수 있습니다.

# 3. Method

---

### Generate outfit(modified) vector

<img width="910" alt="image 2" src="https://github.com/user-attachments/assets/e77bdda7-71cc-4377-88db-8b28f9d1e419">


- 학습을 위해 배경을 흰색으로 날린(By. SAM) 상의, 하의, 신발을 포함한 전신샷 코디이미지를 준비했습니다.
- 코디 이미지를 적절한 비율로 상의, 하의, 신발로 crop하여 코디 내의 단일 이미지를 image encoder가 잘 이해할 수 있도록 했습니다.

<img width="686" alt="image 3" src="https://github.com/user-attachments/assets/32db773f-5f94-4921-89df-e010a7a1d365">


1. 크롭된 코디 이미지(상의, 하의, 신발)를 각각 Image Encoder(FASHION CLIP)을 사용해서 512차원으로 인코딩하였습니다.
2. 각 카테고리 별 인코딩된 벡터들을 concatenate하여 (3, 512)차원의 outfit vector를 만듭니다.
3. outfit vector에서 랜덤한 두개의 카테고리를 지정해(ex. [상의, 하의] 또는 [상의, 신발]등..) 랜덤한 데이터에서 뽑아서 변경하여 modified vector를 만듭니다.
4. outfit vector는 명확한 어울림을 학습할 ‘정답 벡터’, modified vector는 어울림의 정도를 학습할 벡터로 사용할 것입니다.

### Training

<img width="707" alt="image 4" src="https://github.com/user-attachments/assets/ffc65012-6aaf-4d3a-b2a9-f4c99715f5f6">


- 학습은 단일 latent space에서 어울림을 학습합니다
    - outfit vector와 modified vector 모두 같은 latent space에서 동일한 어울림을 학습합니다.
    - 학습할 때에는 각각 카테고리 조합별로 contrastive loss를 수행합니다.  (상의, 하의), (하의, 신발), (상의, 신발) 총 3번의 loss를 수행한 후 이에 대한 평균을 구합니다.
    - outfit vector는 명확한 어울림을 학습하기 위한 데이터로, 각각의 유사도가 최대일 때, loss가 최소가 되도록 하였습니다.
    - modified vector는 적당한 어울림을 학습하기 위한 데이터로, 특정 threshold 이상일때는 어울림, 즉 loss를 최대한 갖지 않도록 하고, 특정 threshold 이하일때는 어울리지 않음, 즉, 큰 loss를 갖도록 하였습니다.
    - 두 벡터에서의 loss를 합한 것이 최종적인 loss가 됩니다.
- overfitting을 방지하기 위해 dropout과 layernorm을 사용했습니다.

# 4. Experiments

---

<img width="620" alt="image 5" src="https://github.com/user-attachments/assets/e162cf55-35d9-4ed8-902d-936d4da4a79f">


- overfitting 없이 학습이 되었습니다.

<img width="894" alt="image 6" src="https://github.com/user-attachments/assets/c2e43cd8-c399-4b85-a9ca-f138e86fa043">


- outfit vector는 유사도가 modified vector(가우시안 분포를 띄는것으로 보입니다)보다 더 크게 분포되어있는 것을 알 수 있습니다.
- outfit vector와 modified vector 간의 유사도를 통해 명확히 어울리는 것은 어울리는 것으로, 어울리지 않는 것은 어울리지 않음으로 처리하는 것을 확인할 수 있습니다.

<img width="610" alt="image 7" src="https://github.com/user-attachments/assets/d1fd2282-352a-403c-bc93-2f341cb43ad9">

- 다만 낮은 threshold에서만 성능이 좋고, threshold가 높아질 수록 성능이 떨어진 것을 확인할 수 있었습니다.

<img width="414" alt="image" src="https://github.com/user-attachments/assets/d473a5f4-57ea-418c-8a7c-cf8293b860b2">

<img width="363" alt="image" src="https://github.com/user-attachments/assets/205b7957-676e-41f3-94b3-b8dde1e3455f">



# 5. Discussion

---

1. 코디 데이터는 단일 패션아이템이 온전히 드러나는 것이 아닌 일부가 가려진 경우가 많습니다. 이를 FASHION CLIP이 명확히 인식하게 하기 위해 data augmentation을 포함한 FASHION CLIP의 finetuning이 필요해보입니다.
2. outfit vector와 modified vector를 동시에 학습시켰기 때문에 modified vector는 온전히 어울림을 학습하지 못한채로 어울림의 정도를 학습할 가능성이 커보입니다. 이를 위해 outfit vector 먼저 학습을 한 후에, modified vector를 학습해보는 방안도 생각 중에 있습니다.
3. outfit vector와 modified vector를 동일한 latent space 내에서 학습을 시켰기 때문에 명확하지 않은 어울림을 학습했을 가능성이 커보입니다. 이를 위해 각 카테고리 별로 다른 임베딩 공간을 두거나 또는 다른 방법을 고려할 예정입니다.
4. 코디 데이터의 품질과 수(무신사 크롤링→SAM거침 / 총 2만3천개)가 문제가 되었을 수도 있을 것 같습니다. 이를 해결할 방안을 모색 중입니다.
