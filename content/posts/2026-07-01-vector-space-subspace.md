---
title: "선형대수학: 벡터 공간과 부분공간, 기저와 Gram-Schmidt 직교화"
date: 2026-07-01
draft: false
tags: ["이어드림스쿨", "선형대수학", "머신러닝", "수업정리", "Python"]
---

## 개요

벡터 공간 안에서 더 작은 구조인 부분공간을 정의하고, 이를 다루기 위한 핵심 개념들(일차결합, 일차독립, 기저, 직교기저)을 정리한다. 마지막으로 임의의 기저를 직교기저로 바꾸는 Gram-Schmidt 과정을 다룬다.

## 1. 부분공간 (Subspace)

**정의**

$W(\neq \varnothing)$이 벡터공간 $R^n$의 부분집합일 때, 다음 두 조건을 만족하면 $W$를 $R^n$의 부분공간이라 한다.

1. 덧셈에 닫혀 있음: $x, y \in W \Rightarrow x + y \in W$
2. 스칼라 곱에 닫혀 있음: $x \in W \Rightarrow \lambda x \in W \ (\forall \lambda \in \mathbb{R})$

부분공간은 반드시 원점(0벡터)을 포함해야 한다 ($\lambda = 0$을 대입하면 성립).

**판별 예시**

```python
# W = { t*(1,2) | t ∈ R } — 원점을 지나는 직선
direction = np.array([1, 2])
t1, t2 = 1.5, -0.5
v1, v2 = t1 * direction, t2 * direction

v_sum = v1 + v2                  # 덧셈 닫힘 확인
t_sum = v_sum[0] / direction[0]  # 다시 t*(1,2) 형태인지 확인

lam = 3.0
v_scaled = lam * v1              # 스칼라 곱 닫힘 확인
```

원점을 지나는 직선과 평면은 부분공간이지만, 원점을 지나는 포물선은 부분공간이 아니다 — $(1,1)$과 $(1,1)$을 더한 $(2,2)$가 곡선 $y=x^2$ 위에 있지 않기 때문에 덧셈에 닫혀 있지 않다.

![부분공간 예시](/images/vector-space/subspace-examples.png)

## 2. 일차결합과 Span

두 벡터 $x_1, x_2$의 일차결합은 $\lambda_1 x_1 + \lambda_2 x_2$ 형태로 표현되며, 기하학적으로는 평행사변형 법칙과 같다. 가능한 모든 계수 조합으로 만들어지는 점들의 집합을 Span이라 하고, 서로 다른 방향의 두 벡터는 $R^2$ 전체를 생성한다.

```python
x1, x2 = np.array([3, 0]), np.array([1, 2])
lam_range = np.linspace(-2, 2, 15)

points = [l1*x1 + l2*x2 for l1 in lam_range for l2 in lam_range]
```

![일차결합과 Span](/images/vector-space/linear-combination.png)

## 3. 일차독립과 일차종속

$\lambda_1 v_1 + \cdots + \lambda_n v_n = 0$을 만족하는 계수가 모두 0인 자명해뿐이면 일차독립, 0이 아닌 해가 존재하면 일차종속이다. 행렬의 rank나 행렬식으로 판별할 수 있다.

```python
def check_linear_independence(vectors):
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    return rank == len(vectors)

# u3 = u1 + u2 인 경우 → 일차종속
u1, u2 = np.array([1,0,0]), np.array([0,1,0])
u3 = u1 + u2
check_linear_independence([u1, u2, u3])  # False
```

평행한 두 벡터는 같은 직선 위에 있어 일차종속이고, 방향이 다른 두 벡터는 일차독립이다.

![일차독립과 일차종속](/images/vector-space/linear-independence.png)

## 4. 기저와 차원

Span이 전체 공간과 같고 일차독립인 벡터 집합을 기저라 하며, 기저를 구성하는 벡터의 개수가 그 공간의 차원이다. 기저가 바뀌면 같은 점이라도 좌표 표현이 달라진다.

```python
# 표준기저가 아닌 새로운 기저로 좌표 변환
b1, b2 = np.array([1, 1]), np.array([-1, 1])
B = np.column_stack([b1, b2])

a, b = 2, 3
coords = np.linalg.solve(B, [a, b])  # 새 기저에서의 좌표
```

두 부분공간의 합에 대한 차원 공식은 다음과 같다.

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$$

## 5. 직교기저와 정규직교기저

기저 벡터들이 서로 직교($\langle u_i, u_j \rangle = 0,\ i \neq j$)하면 직교기저, 여기에 모든 벡터의 노름이 1이면 정규직교기저다. Gram 행렬(내적 행렬)이 대각행렬이면 직교기저, 단위행렬이면 정규직교기저임을 의미한다.

```python
def check_orthogonal_basis(vectors):
    n = len(vectors)
    G = np.array([[np.dot(vectors[i], vectors[j]) for j in range(n)] for i in range(n)])
    is_orthogonal = np.allclose(G - np.diag(np.diag(G)), 0)
    return G, is_orthogonal
```

## 6. Gram-Schmidt 직교화

일차독립인 임의의 벡터 집합을 직교기저로 변환하는 절차다. 각 벡터에서 이전 직교 벡터들에 대한 투영 성분을 순서대로 제거한다.

```python
def gram_schmidt(vectors, normalize=True):
    V = []
    for u in vectors:
        v = u.astype(float).copy()
        proj_sum = np.zeros_like(v)
        for vj in V:
            coeff = np.dot(vj, u) / np.dot(vj, vj)
            proj_sum += coeff * vj
        v = v - proj_sum
        V.append(v)
    if normalize:
        V = [v / np.linalg.norm(v) for v in V]
    return V
```

$u_1=(3,0,0)$, $u_2=(1,2,0)$, $u_3=(1,1,2)$에 적용하면, 서로 직교하는 세 벡터로 변환된다.

![Gram-Schmidt 직교화 과정](/images/vector-space/gram-schmidt.png)

## 7. 직교여공간

부분공간 $W$에 대해, $W$의 모든 벡터와 직교하는 벡터들의 집합을 직교여공간 $W^\perp$이라 한다.

$$R^n = W \oplus W^\perp$$

$W$의 기저 행렬을 SVD로 분해하면 null space가 $W^\perp$의 기저가 된다.

```python
def orthogonal_complement_basis(W_basis):
    W = np.array(W_basis).T
    _, s, Vt = np.linalg.svd(W)
    rank = np.sum(s > 1e-10)
    return Vt[rank:]  # null space = W^⊥ 기저
```

xy평면(기저: $e_1, e_2$)의 직교여공간은 z축이며, $\dim(W) + \dim(W^\perp) = \dim(R^3)$이 성립한다.

## 핵심 요약

| 개념 | 정의 | 핵심 조건 |
|------|------|-----------|
| 부분공간 | $W \subset R^n$, 연산에 닫혀 있음 | 덧셈 닫힘 + 스칼라곱 닫힘 |
| Span | 벡터들의 모든 일차결합 집합 | 자동으로 부분공간 |
| 일차독립 | $\lambda_1 v_1 + \cdots = 0 \Rightarrow$ 모두 0 | 자명해만 존재 |
| 기저 | Span = 전체공간 & 일차독립 | 공간을 최소한으로 생성 |
| 직교기저 | 기저 벡터끼리 내적 = 0 | $\langle u_i, u_j \rangle = 0\ (i \neq j)$ |
| 정규직교기저 | 직교기저 + 단위벡터 | $\langle u_i, u_j \rangle = \delta_{ij}$ |
| Gram-Schmidt | 일반 기저 → 직교기저 | 투영 성분 제거 |
| 직교여공간 | $W$에 직교하는 벡터 전체 | $R^n = W \oplus W^\perp$ |

## 머신러닝과의 연결

- **PCA**: 공분산 행렬의 고유벡터로 만든 직교기저를 이용해 차원을 축소
- **QR 분해**: Gram-Schmidt를 행렬 형태로 표현, 최소제곱법과 선형 회귀에 활용
- **SVD**: 행공간과 열공간의 정규직교기저를 동시에 계산
- **신경망**: 가중치 행렬의 행/열 공간이 부분공간을 이루며, 표현 가능한 변환의 범위를 결정
