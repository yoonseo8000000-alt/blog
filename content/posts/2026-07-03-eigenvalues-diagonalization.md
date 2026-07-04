---
title: "고윳값, 고유벡터, 행렬의 대각화"
date: 2026-07-03
draft: false
tags: ["선형대수", "머신러닝", "이어드림스쿨"]
categories: ["Study"]
---

## Overview

머신러닝을 위한 선형대수학 6강. 고윳값(eigenvalue)과 고유벡터(eigenvector)의 정의부터 시작해서, 행렬의 대각화, 대칭행렬의 Eigen Decomposition, 양의 정부호 행렬(PDM), 행렬의 제곱근, 그리고 스펙트럴 클러스터링까지 정리했다.

**목차**
1. 고윳값, 고유벡터의 정의
2. 행렬의 대각화
3. 대칭 행렬의 대각화 (Eigen Decomposition)
4. 양의 정부호 행렬 (Positive Definite Matrix)
5. 외적과 행렬의 제곱근
6. 스펙트럴 클러스터링 (Spectral Clustering)

---

## 1. 고윳값, 고유벡터의 정의

$n \times n$ 행렬 $A$를 선형변환이라 할 때, 0이 아닌 벡터 $v$가 다음을 만족하면:

$$Av = \lambda v$$

- $\lambda$: 행렬 $A$의 **고윳값(eigenvalue)**
- $v$: 고윳값 $\lambda$에 대한 **고유벡터(eigenvector)**

고유벡터는 행렬 $A$에 의해 변환되어도 **방향이 변하지 않고 크기만 $\lambda$배** 변하는 특별한 벡터다.

```python
# 강의 예제: A = [[1, -8], [1, -5]]
A = np.array([[1, -8], [1, -5]])
v1 = np.array([2, 1])  # 고유벡터 1, λ₁ = -3
v2 = np.array([4, 1])  # 고유벡터 2, λ₂ = -1

print('Av1 =', A @ v1)
print('-3 * v1 =', -3 * v1)
# => Av1 = -3 * v1
```

![고유벡터는 방향이 유지된다](/images/eigenvalues/01_고유벡터_방향_불변_시각화.png)

### 고윳값 구하기: 특성다항식

$n \times n$ 행렬 $A$에 대해:
- **특성다항식**: $f(\lambda) = \det(\lambda I - A)$
- **특성방정식**: $f(\lambda) = \det(\lambda I - A) = 0$

절차는 두 단계다. 먼저 특성방정식을 풀어 고윳값 $\lambda$를 구하고, 각 $\lambda$에 대해 $(\lambda I - A)v = 0$을 풀어 고유벡터 $v$를 구한다.

```python
def char_poly_2x2(A, lam):
    """2x2 행렬의 특성다항식 계산"""
    return (lam - A[0,0]) * (lam - A[1,1]) - A[0,1] * A[1,0]

A_ex = np.array([[1., -8.], [1., -5.]])
eigenvalues, eigenvectors = np.linalg.eig(A_ex)
print('고윳값:', eigenvalues)
```

![특성다항식 f(λ)=0의 해](/images/eigenvalues/02_특성다항식_f_그래프.png)

고윳값에 중근이 나오는 경우도 있는데, 이때는 대응하는 고유벡터가 하나의 방향(scalar 배수 전체)으로 무수히 많이 존재한다.

**고유값 분해를 활용하면 $A^{100}$ 같은 거듭제곱도 쉽게 계산할 수 있다.** $A = PDP^{-1}$이면 $A^{100} = PD^{100}P^{-1}$이고, $D$는 대각행렬이라 $D^{100}$은 대각원소를 각각 100제곱만 하면 되기 때문이다.

---

## 2. 행렬의 대각화

$n \times n$ 행렬 $A$에 대하여, 정칙행렬 $P$가 존재해서 $P^{-1}AP = D$ (D는 대각행렬)를 만족하면 $A$는 **대각화 가능**하다고 한다.

**대각화 가능 여부 판단**: $n$개의 고유벡터 $v_1, \ldots, v_n$이 일차독립이면 $A$를 대각화할 수 있다.

- 고윳값이 모두 다른 경우 → 항상 대각화 가능
- 고윳값에 중근이 있는 경우 → 대각화 가능할 수도, 불가능할 수도 있음 (중근에 대응하는 고유벡터의 rank로 판단)

```python
def diagonalize(A, verbose=True):
    """행렬 A를 대각화하는 함수"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    D_check = P_inv @ A @ P
    print('대각화 성공!:', np.allclose(D_check.real, D.real, atol=1e-10))
    return eigenvalues, P, D
```

중근이 있을 때 대각화 가능 여부는 `rank(λI - A)`로 판단한다. 예를 들어 3×3 행렬에서 중근 $\lambda=2$에 대해 `rank(λI-A) = 2`이면 선형독립인 고유벡터를 2개 구할 수 없어 대각화가 불가능하고, `rank = 1`이면 2개를 구할 수 있어 대각화가 가능하다.

![행렬의 대각화: A = PDP⁻¹](/images/eigenvalues/03_대각화_기하학적_의미.png)

---

## 3. 대칭 행렬의 대각화 (Eigen Decomposition)

$n \times n$ 행렬 $A$가 **대칭행렬** ($A^T = A$)이면, **정규직교행렬** $Q$ ($Q^TQ = I$)가 존재해서:

$$A = Q^T D Q$$

로 항상 대각화할 수 있다. 그리고 대칭행렬에서 **서로 다른 고윳값에 대응하는 고유벡터들은 서로 직교**한다.

```python
A10 = np.array([[1., 2.], [2., 1.]])
eigenvalues10, eigenvectors10 = np.linalg.eigh(A10)  # eigh: 대칭행렬 전용

# 직교성 확인
v1_10 = eigenvectors10[:, 0]
v2_10 = eigenvectors10[:, 1]
dot_product = np.dot(v1_10, v2_10)
print(f'v₁ · v₂ = {dot_product:.10f} ≈ 0 => 직교!')

Q = eigenvectors10
D = np.diag(eigenvalues10)
print('Q^T Q =', np.round(Q.T @ Q, 6))  # => 단위행렬
```

일반 행렬과 대칭 행렬의 가장 큰 차이는 아래와 같다.

| 특성 | 일반 행렬 | 대칭 행렬 |
|---|---|---|
| 실수 고윳값 | 보장 안됨 | 항상 보장 |
| 직교 고유벡터 | 보장 안됨 | 항상 보장 |
| 대각화 가능성 | 조건부 | 항상 가능 |
| 분해 형태 | $P^{-1}AP=D$ | $Q^TAQ=D$ |

![대칭행렬의 고유벡터는 서로 직교한다](/images/eigenvalues/04_대칭행렬_고유벡터_직교성.png)

---

## 4. 양의 정부호 행렬 (Positive Definite Matrix)

대칭행렬 $A$에 대해:

| 종류 | 조건 |
|---|---|
| 양의 정부호 (PDM) | 모든 $x \neq 0$에 대해 $x^TAx > 0$ |
| 양의 준정부호 (PSDM) | 모든 $x$에 대해 $x^TAx \geq 0$ |

다음 네 조건은 모두 동치다.
1. 행렬 $A$가 양의 정부호 행렬
2. $A$의 모든 eigenvalue가 양의 실수
3. 정칙 행렬 $U$가 존재해서 $A = U^TU$
4. $A$의 모든 sub-determinant가 양의 실수

```python
def check_positive_definite(A, name='A'):
    """양의 정부호 행렬 여부 확인"""
    is_symmetric = np.allclose(A, A.T)
    eigenvalues = np.linalg.eigvalsh(A)
    all_positive = np.all(eigenvalues > 0)

    if all_positive:
        print(f'{name}는 양의 정부호 행렬 (PDM)')
    return eigenvalues
```

임의의 행렬 $A$에 대해 $A^TA$, $AA^T$는 **항상 양의 준정부호 행렬**이라는 점도 중요한 성질이다.

![이차형식 x^TAx의 부호 특성](/images/eigenvalues/05_이차형식_x_TAx_시각화.png)

---

## 5. 외적과 행렬의 제곱근

두 벡터 $x, y$의 외적 $x^Ty$는 행렬을 만드는데, 특히 $x^Tx$는 **대칭행렬이자 양의 준정부호 행렬**이다.

행렬의 제곱근은 $B = \sqrt{A} \Leftrightarrow B^2 = A$ (단, $A \geq 0$)로 정의한다. $A$가 PDM이고 $A = PDP^T$로 대각화되면:

$$\sqrt{A} = P\sqrt{D}P^T$$

```python
A_pdm2 = np.array([[4., 2.], [2., 3.]])
eigenvalues_pdm, P_pdm = np.linalg.eigh(A_pdm2)

sqrt_D = np.diag(np.sqrt(eigenvalues_pdm))
sqrt_A = P_pdm @ sqrt_D @ P_pdm.T

# 검증
print('(√A)² = A:', np.allclose(sqrt_A @ sqrt_A, A_pdm2))
```

또한 $AA^T$는 각 열벡터의 외적들의 합 $\sum a_i a_i^T$로도 표현할 수 있다.

![외적과 행렬의 제곱근](/images/eigenvalues/05b_외적과_행렬제곱근.png)

---

## 6. 스펙트럴 클러스터링 (Spectral Clustering)

스펙트럴 클러스터링은 **Laplacian 행렬의 고유벡터**를 이용해 데이터를 클러스터링하는 방법이다.

$$L = D - W$$

- $W$: 인접행렬(adjacency matrix), 노드 간 가중치
- $D$: Degree 행렬(대각행렬)

$$\text{argmin}_f \sum w_{ij}(f_i - f_j)^2 = \text{argmin}_f\ 2[f^TLf]$$

즉 $L$의 고유벡터가 클러스터 분할 정보를 담고 있다. 특히 **두 번째로 작은 고유벡터(Fiedler vector)**의 부호로 그래프를 두 그룹으로 나눌 수 있다.

인접행렬 $W$는 보통 가우시안 커널 $w_{ij} = \exp(-\|x_i-x_j\|^2 / 2\sigma^2)$로 구성한다. 노드가 가까울수록 가중치가 크고, 멀수록 가중치가 작아진다.

![가우시안 커널: 거리가 가까울수록 가중치가 크다](/images/eigenvalues/08_가우시안_커널.png)

```python
# 5개 노드 그래프의 Laplacian
W = np.array([
    [5, 4, 4, 0, 0],
    [4, 5, 4, 0, 0],
    [4, 4, 5, 1, 0],
    [0, 0, 1, 5, 4],
    [0, 0, 0, 4, 5]
], dtype=float)

D_mat = np.diag(W.sum(axis=1))
L = D_mat - W

eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)
fiedler_vector = eigenvectors_L[:, 1]  # 두 번째로 작은 고유벡터

cluster_A = np.where(fiedler_vector > 0)[0] + 1
cluster_B = np.where(fiedler_vector <= 0)[0] + 1
```

동심원이나 반달 모양처럼 **선형으로 분리되지 않는 데이터**에서 K-means는 실패하지만, 스펙트럴 클러스터링은 Laplacian 고유벡터를 활용해 비선형 구조도 잘 분리한다.

![스펙트럴 클러스터링 vs K-means](/images/eigenvalues/06_스펙트럴_vs_K_means.png)

직접 구현해보면 5단계로 정리된다: (1) 가우시안 커널로 인접행렬 $W$ 구성 → (2) Degree 행렬 $D$ 구성 → (3) Laplacian $L = D-W$ 계산 → (4) $L$의 고유벡터 계산 → (5) 상위 고유벡터 공간에서 K-means 클러스터링.

```python
def spectral_clustering_manual(X, n_clusters=2, sigma=1.0):
    n = len(X)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_sq = np.sum((X[i] - X[j])**2)
            W[i, j] = np.exp(-dist_sq / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:n_clusters]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, eigenvalues, eigenvectors
```

![스펙트럴 클러스터링 직접 구현](/images/eigenvalues/07_스펙트럴_클러스터링_직접_구현.png)

---

## 전체 요약

| 개념 | 정의 | 핵심 성질 |
|---|---|---|
| 고윳값/고유벡터 | $Av = \lambda v$ | 방향 불변, 크기만 $\lambda$배 |
| 특성다항식 | $\det(\lambda I - A) = 0$ | 고윳값 찾는 방정식 |
| 대각화 | $P^{-1}AP = D$ | 독립 고유벡터가 n개이면 가능 |
| 대칭행렬 대각화 | $A = Q^TDQ$ | 항상 가능, 고유벡터들이 직교 |
| PDM | $x^TAx > 0$ | 고윳값 모두 양수 ↔ PDM |
| 행렬의 제곱근 | $\sqrt{A} = P\sqrt{D}P^T$ | PDM에서만 정의 |
| 스펙트럴 클러스터링 | $L=D-W$의 고유벡터 | 비선형 클러스터링 가능 |

![Chapter 06 전체 요약](/images/eigenvalues/09_종합_정리_시각화.png)
