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

```python
import numpy as np

vocab = ['"I"', '"love"', '"machine"', '"learning"']
sent1 = np.array([1, 1, 1, 1])   # "I love machine learning"
sent2 = np.array([1, 1, 0, 1])   # "I love deep learning"
```

![Bag-of-Words 벡터 표현](/images/linear-algebra/bow-vector.png)

**이미지 → 행렬**

컬러 이미지는 R, G, B 세 채널의 픽셀 값 행렬로 구성된다. 45×40 크기 이미지는 45×40×3 형태이며, 이를 1차원으로 펼치면(flatten) 5,400차원 벡터가 된다.

```python
img_rgb = np.random.randint(80, 256, (45, 40, 3), dtype=np.uint8)

gray = (0.299*img_rgb[:,:,0] + 0.587*img_rgb[:,:,1] + 0.114*img_rgb[:,:,2]).astype(np.uint8)

print(img_rgb.shape)                    # (45, 40, 3)
print(img_rgb.flatten().shape[0])       # 5400
```

![이미지의 행렬 표현](/images/linear-algebra/image-matrix.png)

## 2. 모델 표현

신경망의 한 층은 다음 식으로 표현된다.

```
u = Wx + b
y = f(u)
```

- `x`: 입력 벡터 / `W`: 가중치 행렬 / `b`: 편향 벡터 / `f`: 활성화 함수

```python
n_in, n_out = 3, 4

x_input = np.array([[1.0], [0.5], [-0.3]])
W = np.random.randn(n_out, n_in).round(2)
b = np.random.randn(n_out, 1).round(2)

u = W @ x_input + b
y_relu = np.maximum(0, u)
```

여러 층을 쌓으면 신경망 구조가 되며, 각 층은 행렬 곱과 비선형 변환의 반복으로 구성된다. 대표적인 활성화 함수 3종은 다음과 같다.

![활성화 함수 비교](/images/linear-algebra/activation-functions.png)

## 3. 행렬과 선형 변환

행렬은 기하학적으로 벡터 공간을 다른 형태로 변환하는 연산으로 해석할 수 있다. 단위 원에 2×2 행렬을 곱하면 원이 타원으로 찌그러지는 것을 확인할 수 있다.

```python
theta = np.linspace(0, 2*np.pi, 300)
circle = np.array([np.cos(theta), np.sin(theta)])

transform_M = np.array([[2., 1.], [0.5, 1.5]])
transformed = transform_M @ circle
```

![행렬을 이용한 선형 변환](/images/linear-algebra/linear-transform.png)

## 4. 차원 축소 (PCA)

고차원 데이터를 저차원으로 압축하는 기법으로, 분산이 가장 큰 방향(주성분)을 찾아 데이터를 투영한다.

**절차**

1. 데이터 행렬 X의 공분산 행렬 계산
2. 공분산 행렬의 고유값 분해 (고유값 크기순 정렬)
3. 상위 k개 고유벡터 방향으로 데이터 투영
4. 각 주성분의 분산 설명 비율(Explained Variance Ratio) 확인

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_3d)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_

for i, e in enumerate(evr):
    print(f"PC{i+1}: {e*100:.1f}%")
```

3차원 데이터를 2차원으로 축소하는 실습에서, 상위 두 개 주성분(PC1, PC2)만으로 원본 데이터 분산의 대부분을 설명할 수 있음을 확인했다.

![PCA를 이용한 차원 축소](/images/linear-algebra/pca-dimension-reduction.png)

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

**선형방정식으로 비즈니스 문제 풀기**

광고 채널 A, B에 대한 예산·시간 제약을 연립방정식으로 세우고, 행렬로 풀어 최적 집행 횟수를 구하는 실습이다.

![선형방정식으로 최적해 구하기](/images/linear-algebra/quiz-linear-equations.png)

제약 조건이 바뀌었을 때(예산·시간 변화) 최적해가 어떻게 이동하는지도 함께 비교했다.

![제약 조건 변화에 따른 시나리오 비교](/images/linear-algebra/quiz-scenario-comparison.png)

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

![선형대수학 학습 로드맵](/images/linear-algebra/curriculum-roadmap.png)

1. **기초**: 벡터·행렬 연산 (덧셈, 뺄셈, 내적, 전치, 역행렬)
2. **심화**: 벡터 공간, 선형 변환, 고유값·고유벡터, 특이값 분해(SVD)
3. **ML 응용**: PCA, 확률론 기초, 미분·경사하강법
4. **DL 응용**: 신경망 구조, 역전파
