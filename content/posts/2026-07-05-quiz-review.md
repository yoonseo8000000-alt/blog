---
title: "확률/미적분 퀴즈 풀이 모음 (베이즈 정리, 조건부 확률·독립, 미분, MVT)"
date: 2026-07-05
draft: false
tags: ["확률통계", "미적분", "머신러닝", "이어드림스쿨"]
categories: ["Study"]
---

## Overview

수업 퀴즈 6개를 풀이와 함께 정리했다. 베이즈 정리, 조건부 확률(유방암 검사 예시), 조건부 독립, 로그함수의 고계도함수, 평균값 정리(MVT), 임계점 계산까지 다룬다. 각 풀이는 수식으로 답을 구한 뒤, 시뮬레이션이나 수치 검증으로 답이 맞는지 다시 확인하는 방식으로 진행했다.

**목차**
1. 베이지안 공식
2. 조건부 확률 (유방암 검사)
3. 조건부 독립식
4. 미분 계산 (고계도함수)
5. 평균값 정리 (MVT)
6. 임계점 계산

---

## 1. 베이지안 공식

베이즈 정리의 빈칸을 채우는 문제로, 정답은 $P(B|A)$다.

$$P(A|B) = \dfrac{P(A)\,P(B|A)}{P(B)}$$

질병 검사 예시(사전확률 $P(A)=0.01$, 민감도 $P(B|A)=0.95$, 위양성률 $P(B|\lnot A)=0.05$)로 100만 명을 시뮬레이션해서 공식과 실제 비율이 일치하는지 검증했다.

```python
np.random.seed(42)
N = 1_000_000
P_A = 0.01
P_B_given_A = 0.95
P_B_given_notA = 0.05

A = np.random.rand(N) < P_A
B = np.where(A,
             np.random.rand(N) < P_B_given_A,
             np.random.rand(N) < P_B_given_notA)

P_A_given_B_sim = (A & B).sum() / B.sum()

P_B_formula = P_B_given_A * P_A + P_B_given_notA * (1 - P_A)
P_A_given_B_formula = (P_A * P_B_given_A) / P_B_formula
# Simulation: 0.16092, Formula: 0.16102 — 거의 일치
```

![베이즈 정리: 사전확률 vs 사후확률, 시뮬레이션 vs 공식](/images/quiz/01_bayes_proof.png)

---

## 2. 조건부 확률 (유방암 검사)

$P(\text{유방암})=0.004$, $P(\text{양성}|\text{유방암})=0.9$, $P(\text{양성}|\text{無유방암})=0.1$일 때:

$$P(\text{양성}) = 0.9\times0.004 + 0.1\times0.996 = 0.1032$$
$$P(\text{유방암}|\text{양성}) = \frac{0.0036}{0.1032} \approx 0.0349$$

반올림하면 **0.035**가 정답이다. 몬테카를로 시뮬레이션(200만 건)으로도 0.03493으로 거의 동일하게 나왔다.

```python
P_disease = 0.004
P_pos_given_disease = 0.90
P_pos_given_no_disease = 0.10

P_no_disease = 1 - P_disease
P_positive = (P_pos_given_disease * P_disease) + (P_pos_given_no_disease * P_no_disease)
P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive
# 0.0349
```

10만 명을 기준으로 True Positive, False Negative, False Positive, True Negative 인원수를 나눠보면 왜 양성 판정을 받아도 실제 질병 확률이 낮은지(위양성이 훨씬 많기 때문) 직관적으로 확인할 수 있다.

![유방암 검사 베이즈 정리: 공식 vs 시뮬레이션, 인구 10만 명 분류](/images/quiz/02_breast_cancer_bayes.png)

---

## 3. 조건부 독립식

조건부 독립이면 $p(x,y|z) = p(x|z)\,p(y|z)$가 성립한다. $z$ 값(0 또는 1)마다 $x$와 $y$를 서로 독립적으로 생성한 뒤, 실제 결합확률과 각 확률의 곱이 일치하는지 300만 건 시뮬레이션으로 확인했다.

```python
P_z = np.array([0.4, 0.6])
P_x_given_z = np.array([0.7, 0.2])
P_y_given_z = np.array([0.3, 0.8])

z = (np.random.rand(N) < P_z[1]).astype(int)
x = (np.random.rand(N) < P_x_given_z[z]).astype(int)
y = (np.random.rand(N) < P_y_given_z[z]).astype(int)

# z=0: empirical p(x,y|z)=0.21009, p(x|z)*p(y|z)=0.21013 (차이 0.00004)
# z=1: empirical p(x,y|z)=0.16029, p(x|z)*p(y|z)=0.16024 (차이 0.00005)
```

$z$의 두 값 모두에서 실제 결합확률과 곱이 오차 0.0001 이내로 일치해서, 조건부 독립 관계가 성립함을 확인했다.

![조건부 독립: p(x,y|z) = p(x|z)·p(y|z) 검증](/images/quiz/03_conditional_independence.png)

---

## 4. 미분 계산 (고계도함수)

$f(x) = \ln(1+x)$의 4차 미분을 구하는 문제다.

$$f'(x) = \frac{1}{1+x}, \quad f''(x) = -\frac{1}{(1+x)^2}, \quad f'''(x) = \frac{2}{(1+x)^3}, \quad f^{(4)}(x) = -\frac{6}{(1+x)^4} = -\frac{3!}{(1+x)^4}$$

부호가 매번 바뀌고 계수는 $(n-1)!$ 형태로 누적된다. 정답은 $-\dfrac{3!}{(1+x)^4}$다.

해석적으로 구한 4차 도함수를, 중심차분(central finite difference) 방식의 수치 미분과 비교해서 검증했다.

```python
def f4_analytical(x):
    return -6 / (1 + x)**4

def numerical_derivative(func, x, h=1e-2):
    """4차 중심차분"""
    return (f(x-2*h) - 4*f(x-h) + 6*f(x) - 4*f(x+h) + f(x+2*h)) / h**4

# x=1: numerical=-0.37503, analytical=-0.37500, diff=0.00003
```

두 곡선(해석적 도함수, 수치 미분)이 거의 겹치고, 오차도 매우 작다.

![f(x)=ln(1+x)의 4차 도함수: 해석적 vs 수치 미분](/images/quiz/04_derivative_proof.png)

---

## 5. 평균값 정리 (MVT)

함수 $f(x)$가 $[a,b]$에서 연속이고 $(a,b)$에서 미분가능하면, 평균변화율과 같은 순간변화율을 갖는 점 $c$가 반드시 존재한다.

$$\frac{f(b)-f(a)}{b-a} = f'(c), \quad (a<c<b)$$

기하학적으로는 두 점을 잇는 **할선(secant line)의 기울기**와 같은 기울기를 가지는 **접선**이 구간 내부에 존재한다는 의미다.

$f(x)=x^3-3x^2+2x+5$, $[a,b]=[0.2, 2.5]$ 구간으로 검증하면 할선 기울기는 0.69이고, 이 값과 같은 접선 기울기를 갖는 $c$가 구간 내부에 **0.249와 1.751 두 곳**에서 발견됐다 (둘 다 $a<c<b$ 만족).

```python
def f(x): return x**3 - 3*x**2 + 2*x + 5
def f_prime(x): return 3*x**2 - 6*x + 2

secant_slope = (f(b) - f(a)) / (b - a)  # 0.69

# f'(c) = secant_slope를 만족하는 c를 구간 내에서 탐색 (brentq)
# c = 0.24944, c = 1.75056 — 둘 다 f'(c) = 0.69로 일치
```

할선과 두 접선이 평행하게 그려지는 걸 시각적으로 확인할 수 있다.

![평균값 정리: 할선과 평행한 접선이 존재하는 두 지점](/images/quiz/05_mvt_proof.png)

---

## 6. 임계점 계산

$f(x) = \dfrac{2x}{x^2+1}$의 임계점(도함수가 0이 되는 지점)을 구하는 문제다. 몫의 미분법을 적용하면:

$$f'(x) = \frac{2(x^2+1) - 2x\cdot2x}{(x^2+1)^2} = \frac{2-2x^2}{(x^2+1)^2}$$

$f'(x)=0$이 되려면 분자가 0이어야 하므로:

$$2-2x^2=0 \;\Rightarrow\; x^2=1 \;\Rightarrow\; x=\pm1$$

**정답: $x=-1, 1$**이다. 후보값들을 하나씩 대입해서 해석적 도함수와 수치 미분이 모두 0에 가까운지 확인하고, $-10$부터 $10$까지 전 구간을 스캔해서 부호가 바뀌는 지점을 찾아도 정확히 $x=-1, 1$만 나왔다.

```python
def f_prime_analytic(x):
    return (2 - 2*x**2) / (x**2 + 1)**2

# x=-1: f'(x) = 0.000000  <-- critical point
# x= 1: f'(x) = 0.000000  <-- critical point
```

$f(x)$ 그래프에서 극댓값·극솟값이 나타나는 지점과, $f'(x)$가 0을 지나는 지점이 정확히 일치한다.

![f(x) = 2x/(x²+1)의 임계점: x = -1, 1](/images/quiz/06_critical_points.png)
