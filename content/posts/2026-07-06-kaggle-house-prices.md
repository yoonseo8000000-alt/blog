---
title: "Kaggle House Prices: XGBoost/LightGBM 앙상블로 주택 가격 예측하기"
date: 2026-07-06
draft: false
tags: ["Kaggle", "머신러닝", "회귀분석", "이어드림스쿨"]
categories: ["Study"]
---

## Overview

Kaggle의 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 대회를 다룬 회귀 프로젝트입니다. 주택의 79개 특성(면적, 위치, 건축 연도, 상태 등)을 활용해 최종 판매 가격(SalePrice)을 예측했습니다.

전체 코드는 [GitHub 레포](https://github.com/yoonseo8000000-alt/kaggle-house-prices)에서 확인할 수 있습니다.

## Problem

- 결측치가 다양한 유형으로 존재 (범주형/수치형, 그리고 "결측 = 해당 항목 없음"을 의미하는 경우도 포함)
- 타겟 변수(SalePrice)가 우측으로 편향된(right-skewed) 분포를 가짐
- 다수의 범주형 변수를 어떤 방식으로 인코딩할지 결정 필요

## Approach

### 1. 결측치 처리
학습 데이터와 테스트 데이터를 합쳐서 일관되게 처리했습니다. 단순히 평균/최빈값으로 채우지 않고, 컬럼별 의미에 따라 다르게 처리했습니다 (예: "차고 없음"을 의미하는 결측치는 0 또는 "None"으로).

### 2. 특성 엔지니어링
`TotalSF`(총 면적) 등 도메인 지식을 반영한 파생 변수를 추가로 생성했습니다.

### 3. 이상치 제거
**학습 데이터에서만** 이상치를 제거했습니다. 테스트 데이터는 건드리지 않아야 data leakage를 방지할 수 있기 때문입니다.

### 4. 범주형 인코딩
Label Encoding을 적용한 후 학습/테스트 세트를 분리했습니다.

### 5. 모델 앙상블
세 가지 모델을 학습시키고 비교했습니다.
- XGBoost + LightGBM + RandomForest (3종 앙상블)
- XGBoost + LightGBM (2종 앙상블)

### 6. 하이퍼파라미터 튜닝
XGBoost 기준으로 `learning_rate`를 낮추고(0.05 → 0.01), `n_estimators`를 늘리는(1000 → 2000) 방향으로 튜닝했습니다. 그 외 `max_depth`, `subsample`, `colsample_bytree`, `gamma`도 함께 조정했습니다.

## Results

| 모델 | RMSE | R² |
|---|---|---|
| XGBoost + LightGBM + RandomForest | 28,331 달러 | 0.8954 |
| XGBoost + LightGBM | 28,032 달러 | 0.8976 |
| XGBoost (튜닝) | **27,366 달러** | **0.9024** |

## Learnings

- 이상치 제거는 반드시 학습 데이터에만 적용해야 하며, 테스트 데이터에 적용하면 data leakage로 이어질 수 있습니다.
- 모델을 많이 앙상블한다고 무조건 성능이 좋아지는 것은 아니며, 조합별로 실제 비교가 필요합니다.
- 하이퍼파라미터 튜닝 시 `learning_rate`를 낮추는 대신 `n_estimators`를 늘리는 트레이드오프를 확인했습니다.
- 튜닝된 XGBoost 단일 모델(RMSE 27,366)이 3종 앙상블(RMSE 28,331)보다 더 나은 성능을 보였습니다. 앙상블이 항상 단일 모델보다 우수한 것은 아니며, 개별 모델의 튜닝 수준에 따라 결과가 달라질 수 있음을 확인했습니다.

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `matplotlib` · `seaborn`
