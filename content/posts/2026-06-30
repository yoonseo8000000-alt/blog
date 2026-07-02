---
title: "머신러닝을 위한 선형대수학 기초 정리: 벡터, 행렬, 그리고 PCA까지"
date: 2026-06-30
draft: false
tags: ["이어드림스쿨", "선형대수학", "머신러닝", "수업정리", "Python"]
---

## 개요

머신러닝과 딥러닝에서 데이터와 모델은 대부분 벡터와 행렬로 표현된다. 이번 수업에서는 데이터 표현, 모델 표현, 차원 축소라는 세 가지 관점에서 선형대수학의 쓰임을 다뤘다.

## 1. 데이터 표현

**텍스트 → 벡터 (Bag of Words)**

문장을 고정 길이 숫자 벡터로 변환하는 기본 방법. 어휘 목록을 기준으로 단어의 등장 여부를 1과 0으로 표시한다.

- 어휘: ["I", "love", "machine", "learning"]
- "I love machine learning" → [1, 1, 1, 1]
- "I love deep learning" → [1, 1, 0, 1]

```python
import numpy as np

vocab = ['"I"', '"love"', '"machine"', '"learning"']
sent1 = np.array([1, 1, 1, 1])   # "I love machine learning"
sent2 = np.array([1, 1, 0, 1])   # "I love deep learning"
```

**이미지 → 행렬**

컬러 이미지는 R, G, B 세 채널의 픽셀 값 행렬로 구성된다. 45×40 크기 이미지는 45×40×3 형태이며, 이를 1차원으로 펼치면(flatten) 5,400차원 벡터가 된다.

```python
img_rgb = np.random.randint(80, 256, (45, 40, 3), dtype=np.uint8)

# 그레이스케일 변환 (ITU-R BT.601 가중합)
gray = (0.299*img_rgb[:,:,0] + 0.587*img_rgb[:,:,1] + 0.114*img_rgb[:,:,2]).astype(np.uint8)

print(img_rgb.shape)                    # (45, 40, 3)
print(img_rgb.flatten().shape[0])       # 5400
```

## 2. 모델 표현

신경망의 한 층은 다음 식으로 표현된다.

```
u = Wx + b
y = f(u)
```

- `x`: 입력 벡터 / `W`: 가중치 행렬 / `b`: 편향 벡터 / `f`: 활성화 함수

```python
n_in, n_out = 3, 4

x_input = np.array([[1.0], [0.5], [-0.3]])       # 입력 벡터 (3×1)
W = np.random.randn(n_out, n_in).round(2)         # 가중치 행렬 (4×3)
b = np.random.randn(n_out, 1).round(2)            # 편향 벡터 (4×1)

u = W @ x_input + b        # 선형 결합
y_relu = np.maximum(0, u)  # ReLU 활성화
```

여러 층을 쌓으면 신경망 구조가 되며, 각 층은 행렬 곱과 비선형 변환의 반복으로 구성된다.

## 3. 차원 축소 (PCA)

고차원 데이터를 저차원으로 압축하는 기법으로, 분산이 가장 큰 방향(주성분)을 찾아 데이터를 투영한다.

**절차**

1. 데이터 행렬 X의 공분산 행렬 계산
2. 공분산 행렬의 고유값 분해 (고유값 크기순 정렬)
3. 상위 k개 고유벡터 방향으로 데이터 투영
4. 각 주성분의 분산 설명 비율(Explained Variance Ratio) 확인

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_3d)   # 평균 0, 분산 1로 표준화

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_               # 각 주성분의 분산 설명 비율

for i, e in enumerate(evr):
    print(f"PC{i+1}: {e*100:.1f}%")
```

3차원 데이터를 2차원으로 축소하는 실습에서, 상위 두 개 주성분(PC1, PC2)만으로 원본 데이터 분산의 대부분을 설명할 수 있음을 확인했다.

## 퀴즈 풀이 요약

**내적과 노름(Norm)**

두 벡터의 내적은 대응하는 성분끼리 곱한 뒤 합산한 값이고, 노름은 각 성분의 제곱합에 제곱근을 취한 값이다.

```python
x = np.array([1, -1, 0, 2, 4])
y = np.array([2, 1, 1, 3, 0])

inner = np.dot(x, y)                  # 내적: 7
norm_x = np.linalg.norm(x)            # 노름: √22
norm_y = np.linalg.norm(y)            # 노름: √15
```

**내적의 선형성**

내적은 분배법칙이 성립한다. `<v1, w> = 3`, `<v2, w> = -2`일 때 `<2v1 - 3v2, w>`를 계산하면:

```python
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
w = np.array([3, -2, 0])

result = np.dot(2*v1 - 3*v2, w)
# 분배법칙으로 검증
result_check = 2*np.dot(v1, w) - 3*np.dot(v2, w)

print(result, result_check)  # 12 12
```

**행렬 곱셈**

첫 번째 행렬의 행과 두 번째 행렬의 열을 내적하는 연산이다. A가 m×r, B가 r×n 크기일 때만 곱셈이 정의되며, 결과는 m×n 크기다.

```python
A = np.array([[-1, 2], [-2, 1], [3, 4]])   # 3×2
B = np.array([[1, 0, 2], [2, -3, 1]])       # 2×3

result = np.dot(A, B)   # 3×3
print(result)
# [[  3  -6   0]
#  [  0  -3  -3]
#  [ 11 -12  10]]
```

주요 성질:
- 교환법칙 불성립: AB ≠ BA
- 영행렬이 아닌 두 행렬의 곱이 영행렬이 될 수 있음
- 두 행렬의 곱이 단위행렬(I)이면 서로 역행렬 관계

## 커리큘럼 로드맵

1. **기초**: 벡터·행렬 연산 (덧셈, 뺄셈, 내적, 전치, 역행렬)
2. **심화**: 벡터 공간, 선형 변환, 고유값·고유벡터, 특이값 분해(SVD)
3. **ML 응용**: PCA, 확률론 기초, 미분·경사하강법
4. **DL 응용**: 신경망 구조, 역전파
