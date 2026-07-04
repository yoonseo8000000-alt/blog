---
title: "머신러닝을 위한 선형대수학 02. 선형 변환 (Linear Transformation)"
date: 2026-07-02
draft: false
tags: ["선형대수", "머신러닝", "이어드림스쿨"]
categories: ["study"]
---

## 목차

1. 행렬을 보는 3가지 관점
2. 벡터공간으로써의 행렬 연산
3. 행렬의 크기 — Frobenius Norm
4. 특수 행렬 (대각, 단위, 대칭, 직교행렬)
5. 역행렬
6. 선형변환
7. 행렬의 Rank
8. 행렬식 (Determinant)
9. 직교행렬의 함수적 성질
10. 신경망에서의 선형 변환 응용

---

## 1. 행렬을 보는 3가지 관점

행렬은 크게 **3가지 관점**으로 바라볼 수 있다.

### 관점 1 — 벡터공간의 확장

벡터는 행 또는 열로 배치된 숫자 묶음이며, 크기가 같은 벡터끼리만 연산이 가능하다.
행렬은 이 개념을 2차원으로 확장한 것이다.

$$
A = \begin{pmatrix} 1 & 2 & -1 \\ 3 & 2 & -1 \end{pmatrix}, \quad
B = \begin{pmatrix} 3 & 5 & -7 \\ 0 & 4 & 1 \end{pmatrix}
$$

```python
import numpy as np

A = np.array([[1, 2, -1], [3, 2, -1]])
B = np.array([[3, 5, -7], [0, 4, 1]])
k = 3

print(f"A + B =\n{A + B}")
print(f"{k}A =\n{k * A}")
```

![행렬 덧셈 시각화](/images/vector-space/01-matrix-addition.png)

---

## 2. 벡터공간으로써의 행렬 연산 (정리 5.1)

$A = (a_{ij}),\ B = (b_{ij}) \in M_{m \times n},\ k \in \mathbb{R}$ 일 때:

- **덧셈:** $A + B = (a_{ij} + b_{ij})$
- **스칼라 곱:** $kA = (ka_{ij})$

| 법칙 | 식 |
|------|----|
| 교환법칙 | $A + B = B + A$ |
| 결합법칙 | $(A+B)+C = A+(B+C)$ |
| 항등원 | $A + \mathbf{0} = A$ |
| 역원 | $A + (-A) = \mathbf{0}$ |

```python
rng = np.random.default_rng(42)
A = rng.integers(-5, 6, (3, 3)).astype(float)
B = rng.integers(-5, 6, (3, 3)).astype(float)

print("교환법칙:", np.allclose(A + B, B + A))
```

---

## 3. 행렬의 크기 — Frobenius Norm

행렬도 하나의 벡터이므로 **벡터처럼 크기**를 정의할 수 있다.

$$
\|A\|_F = \left(\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2\right)^{\frac{1}{2}}
$$

```python
A = np.array([[2, -1, 5], [0, 2, 1], [3, 1, 1]], dtype=float)

frob_manual = np.sqrt(np.sum(A**2))
frob_numpy = np.linalg.norm(A, 'fro')

print(f"||A||_F = {frob_manual:.6f}")
```

![Frobenius Norm 비교](/images/vector-space/02-frobenius-norm.png)

---

## 4. 특수 행렬

### 전치행렬 (정의 5.3)

$(i,j)$ 성분을 $(j,i)$ 성분으로 바꾼 행렬. 연산법칙:

$$
(AB)^\top = B^\top A^\top
$$

> ⚠️ 순서가 바뀐다는 점에 주의!

### 대각행렬 / 단위행렬 / 대칭행렬 / 직교행렬

```python
I3 = np.eye(3)
D  = np.diag([2, 3, -2])
S  = np.array([[1, 2, 4], [2, 1, 3], [4, 3, 1]])  # 대칭

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])  # 직교 (회전)
```

![특수 행렬 한눈에 보기](/images/vector-space/03-special-matrices.png)

---

## 5. 역행렬 (정리 5.7)

$A, B$가 정칙행렬일 때:

| 성질 | 식 |
|------|----|
| 곱의 역행렬 | $(AB)^{-1} = B^{-1}A^{-1}$ |
| 스칼라 곱 | $(kA)^{-1} = \frac{1}{k}A^{-1}$ |
| 전치의 역 | $(A^\top)^{-1} = (A^{-1})^\top$ |

```python
A = np.array([[3, 1, 0], [1, 2, 1], [-1, 2, 3]], dtype=float)
A_inv = np.linalg.inv(A)

print(f"A @ A⁻¹ =\n{np.round(A @ A_inv, 10)}")

B = np.array([[2, 0, 1], [1, 3, 0], [0, 1, 2]], dtype=float)
print("(AB)⁻¹ = B⁻¹A⁻¹?", np.allclose(np.linalg.inv(A@B), np.linalg.inv(B) @ np.linalg.inv(A)))
```

![역행렬 검증](/images/vector-space/04-inverse-matrix.png)

---

## 6. 선형변환 (Linear Transformation)

$m \times n$ 행렬 $A$는 벡터공간 $\mathbb{R}^n$에서 $\mathbb{R}^m$으로의 **함수**이며, 다음 선형법칙을 만족한다:

$$
A(\boldsymbol{x}+\boldsymbol{y}) = A\boldsymbol{x} + A\boldsymbol{y}, \qquad
A(\lambda\boldsymbol{x}) = \lambda A\boldsymbol{x}
$$

```python
# 단위원을 변환하는 3가지 예시
transforms = [
    (np.array([[2, 0], [0, 1]]),    '스케일링 (x축 2배)'),
    (np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],
               [np.sin(np.pi/3),  np.cos(np.pi/3)]]), '회전 (60°)'),
    (np.array([[1, 1], [0, 1]]),    '전단 변환 (Shear)'),
]
```

![선형변환 단위원 변환](/images/vector-space/05-linear-transform-circle.png)

---

## 7. 행렬의 Rank

$m \times n$ 행렬 $A$에서 행벡터(또는 열벡터) 중 **일차독립인 벡터의 최대 개수**.

$$
n = \dim\{\boldsymbol{x} \in \mathbb{R}^n \mid A\boldsymbol{x} = \boldsymbol{0}\} + \text{rank}\,A
$$

```python
A = np.array([[-1, 2, -2], [3, 4, -5]], dtype=float)
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

B = np.array([[2, 0, 0], [-5, 1, 2], [3, 8, -7]], dtype=float)
print(f"det(B) = {np.linalg.det(B):.2f}")
print(f"rank(B) = {np.linalg.matrix_rank(B)}")
```

![행렬의 Rank 비교](/images/vector-space/06-rank-comparison.png)

---

## 8. 행렬식 (Determinant)

### 2×2

$$
\det(A) = a_{11}a_{22} - a_{12}a_{21}
$$

### 3×3 (Sarrus 법칙)

$$
\det(A) = [a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32}] - [a_{13}a_{22}a_{31} + a_{11}a_{23}a_{32} + a_{12}a_{21}a_{33}]
$$

> Sarrus 법칙은 $n = 4$ 이상에서는 성립하지 않는다.

**주요 성질:**

$$
\det A = \det A^\top, \qquad
\det(AB) = \det(A)\det(B), \qquad
\det(\lambda A) = \lambda^n \det(A)
$$

```python
# det(A) = 0이 되는 λ 찾기
# A = [[λ+1, 2, -2], [-4, λ-3, 4], [0, 2, λ-1]]
# det = (λ+1)(λ-1)(λ-3) = 0 → λ = 1, -1, 3

lambdas = np.linspace(-2, 4, 500)
dets = [np.linalg.det(np.array([[l+1, 2, -2], [-4, l-3, 4], [0, 2, l-1]])) for l in lambdas]
```

![행렬식 시각화](/images/vector-space/07-determinant.png)

---

## 9. 직교행렬의 함수적 성질 (정리 5.18)

$A : \mathbb{R}^n \to \mathbb{R}^n$ 일 때 다음 세 성질은 **동치**이다:

1. $A$는 직교행렬이다.
2. $\|A\boldsymbol{x}\| = \|\boldsymbol{x}\|$ (크기 보존)
3. $\langle A\boldsymbol{x}, A\boldsymbol{y} \rangle = \langle \boldsymbol{x}, \boldsymbol{y} \rangle$ (내적 보존)

직교행렬은 **두 벡터 사이의 각을 보존**한다.

```python
theta_rot = np.pi / 3
Q = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
              [np.sin(theta_rot),  np.cos(theta_rot)]])

x = np.array([2.0, 1.0])
y = np.array([0.5, 2.0])
Qx, Qy = Q @ x, Q @ y

print("길이 보존:", np.allclose(np.linalg.norm(x), np.linalg.norm(Qx)))
```

![직교행렬 성질 시각화](/images/vector-space/08-orthogonal-matrix.png)

---

## 10. 신경망에서의 선형 변환 응용

입력벡터 $\boldsymbol{x}$, 가중치 행렬 $W$, 편향벡터 $\boldsymbol{b}$가 주어질 때:

$$
\boldsymbol{u} = W \cdot \boldsymbol{x} + \boldsymbol{b}
$$

```python
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

n_in, n_hidden, n_out = 4, 6, 3
x = np.array([1.0, -0.5, 0.8, -1.2])
W1 = np.random.randn(n_hidden, n_in) * 0.5
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_out, n_hidden) * 0.5
b2 = np.zeros(n_out)

# Forward pass
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
output = sigmoid(z2)
```

![신경망 선형변환 시뮬레이션](/images/vector-space/09-neural-network.png)

---

## 핵심 요약

| 개념 | 정의/공식 | 핵심 포인트 |
|------|-----------|-------------|
| 행렬의 3가지 관점 | 확장·함수·변형 | 목적에 따라 관점 전환 |
| Frobenius Norm | $\|A\|_F = (\sum a_{ij}^2)^{1/2}$ | 행렬 = 벡터처럼 크기 측정 |
| 전치행렬 | $A^\top=(a_{ji})$ | $(AB)^\top=B^\top A^\top$ (순서 반전!) |
| 직교행렬 | $A^\top A = I$ | $A^\top = A^{-1}$, 회전·대칭 변환 |
| 역행렬 | $AA^{-1}=I$ | $\det A \neq 0$ iff 역행렬 존재 |
| 선형변환 | $A(x+y)=Ax+Ay$ | 행렬 = 두 벡터공간 사이 함수 |
| Rank | 일차독립인 행/열벡터 수 | $n = \dim N(A) + \text{rank}\,A$ |
| 행렬식 | $\det A$ | $\det A = 0 \Leftrightarrow$ 역행렬 존재 X |
