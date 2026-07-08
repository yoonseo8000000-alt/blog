---
title: "분류 모델의 핵심은 모델이 아니라 검증 설계다 - Pipeline과 하이퍼파라미터 튜닝"
date: 2026-07-08
categories: ["Study"]
tags: ["scikit-learn", "RandomForest", "Pipeline", "GridSearchCV", "Classification"]
---

## Overview

Titanic 데이터셋으로 생존 여부(`survived`)를 예측하는 이진 분류 모델을 만들었다. 이번 실습(Lv4)의 목표는 "더 좋은 모델"을 찾는 게 아니라, **분류 문제에서 데이터 누수 없이 검증을 설계하는 흐름**을 익히는 것이었다.

- 데이터: `seaborn.load_dataset('titanic')`
- 모델: `RandomForestClassifier`
- 핵심 도구: `Pipeline` + `ColumnTransformer` + `StratifiedKFold` + `GridSearchCV`

> 참고: 이번 실습에는 시계열(time series) 관련 코드는 포함되어 있지 않다. 순수 정적(static) 분류 문제였다.

## Problem

분류 문제, 특히 Titanic처럼 클래스 비율이 한쪽으로 치우친 데이터(생존 38% vs 사망 62%)에서는 두 가지를 항상 조심해야 한다.

1. **train/test, 그리고 CV의 각 fold에서 클래스 비율이 흔들리면 안 된다.** 우연히 어떤 fold에 생존자가 몰리면 성능 평가 자체가 왜곡된다.
2. **전처리가 검증 데이터 정보를 미리 알아버리면 안 된다 (data leakage).** 예를 들어 결측치를 전체 데이터의 중앙값으로 채운 뒤 train/test를 나누면, test 데이터 정보가 이미 학습 과정에 스며든 것이다.

이 두 가지를 동시에 해결하는 게 이번 실습의 진짜 목표였다.

## Approach

### 1) 층화추출(Stratify)로 시작

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
```

`stratify=y`를 넣는 것만으로 train/test의 생존 비율이 원본과 거의 동일하게 유지된다. 회귀에서는 신경 쓸 필요 없던 부분인데, 분류에서는 필수다.

### 2) Pipeline으로 전처리 누수 원천 차단

여기가 이번 실습에서 가장 중요하다고 느낀 부분이다.

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
```

**왜 이 구조가 중요한가?**

![Pipeline이 데이터 누수를 막는 구조](/images/titanic-classification-pipeline-tuning/pipeline_leakage_diagram.png)

전처리(imputer, scaler, encoder)를 `Pipeline` 안에 넣으면, 교차검증의 각 fold마다 전처리기가 **오직 그 fold의 학습 데이터에만** `fit`된다. 즉:

- fold 1의 검증 데이터를 채울 중앙값은 fold 1의 학습 데이터에서만 계산됨
- fold 2로 넘어가면 그 중앙값은 버려지고 fold 2의 학습 데이터로 새로 계산됨

이렇게 하지 않고 전처리를 CV 밖에서 미리 해버리면, 검증 fold의 통계치(중앙값, 최빈값 등)가 이미 전체 데이터 계산에 섞여 들어가서 **검증 점수가 실제보다 낙관적으로 부풀려진다.** Pipeline은 이 실수를 구조적으로 막아준다.

수치형은 `median` 대체 + `StandardScaler`, 범주형은 `most_frequent` 대체 + `OneHotEncoder(drop='first')`로 처리했다. `drop='first'`는 더미 변수 간 다중공선성을 줄이기 위한 선택이다.

### 3) StratifiedKFold로 교차검증

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc', 'recall': 'recall'},
    return_train_score=True,
    n_jobs=-1,
)
```

일반 `KFold`가 아니라 `StratifiedKFold`를 쓴 이유는 각 fold에서도 생존 비율을 원본과 비슷하게 유지하기 위해서다. 실제로 fold별 생존 비율을 뽑아보면 0.3803~0.3873 사이로, 전체 비율(약 0.38)과 거의 차이가 없었다 — 층화추출이 의도대로 작동한 걸 직접 확인할 수 있었다.

### 4) GridSearchCV로 하이퍼파라미터 튜닝

이번 실습에서 두 번째로 강조하고 싶은 부분이다.

```python
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, None],
    'model__min_samples_leaf': [1, 3, 5],
    'model__max_features': ['sqrt', 'log2'],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)
```

여기서 눈여겨볼 점 두 가지:

1. **파라미터 이름이 `model__n_estimators`처럼 `__`로 연결된다.** `pipeline`의 `model` 스텝 안에 있는 `RandomForestClassifier`의 `n_estimators`를 지정하려면 이렇게 스텝 이름과 파라미터 이름을 이어줘야 한다. `Pipeline` 전체를 `GridSearchCV`에 넣었기 때문에 가능한 문법이다.
2. **튜닝 대상은 학습 데이터(`X_train`, `y_train`)뿐이다.** 테스트 데이터는 이 단계에서 절대 등장하지 않는다.

튜닝한 4개 하이퍼파라미터의 의미:

| 파라미터 | 역할 |
|---|---|
| `n_estimators` | 만들 트리(나무)의 개수. 많을수록 안정적이지만 느려짐 |
| `max_depth` | 각 트리가 뻗어나갈 수 있는 최대 깊이. 너무 깊으면 과적합 |
| `min_samples_leaf` | leaf 노드가 가져야 할 최소 샘플 수. 클수록 과적합 방지 |
| `max_features` | 각 분기(split)에서 고려할 feature 개수. 트리 간 다양성 조절 |

### 5) 마지막에만 테스트 데이터 사용

```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_proba)
```

`grid_search.best_estimator_`는 최적 파라미터로 **학습 데이터 전체를 다시 학습한 모델**이다. 테스트 데이터는 이 최종 모델의 성능을 딱 한 번 확인하는 용도로만 사용했다.

## Results

Confusion Matrix와 `classification_report`로 정밀도/재현율/F1을 확인했다.

![Titanic 생존 예측 Confusion Matrix](/images/titanic-classification-pipeline-tuning/confusion_matrix.png)

`feature_importances_`로 어떤 변수가 생존 예측에 크게 기여했는지도 확인했다. One-hot 인코딩 이후의 컬럼명은 `get_feature_names_out()`으로 매핑했다.

![RandomForest Feature Importance Top 15](/images/titanic-classification-pipeline-tuning/feature_importance.png)


- fold별 생존 비율을 표로 직접 검증해서 StratifiedKFold가 제대로 작동했는지 눈으로 확인

## Learnings

- **Pipeline은 편의 도구가 아니라 데이터 누수 방지 장치다.** 전처리와 모델을 하나로 묶으면 "이번 fold에서만 fit"이 자동으로 보장된다.
- **분류에서는 stratify가 기본값이어야 한다.** train_test_split이든 CV든, 클래스 비율이 흔들리면 그 위에서 나온 모든 지표가 흔들린다.
- **하이퍼파라미터 튜닝은 파이프라인 문법(`model__파라미터명`)에 익숙해져야 자유롭게 확장할 수 있다.** 나중에 전처리 파라미터(예: `preprocessor__num__imputer__strategy`)까지 grid에 넣는 것도 같은 원리다.
- 이번 실습은 시계열이 아니라 정적 분류였지만, 다음 단계인 BARAM 2026(풍력발전량 예측)에서는 시간 순서를 지키는 `TimeSeriesSplit` 기반 검증이 필요하다 — 여기서 배운 "누수 방지" 철학은 그대로 이어지지만, fold를 나누는 방식 자체는 달라져야 한다.
- 다음으로 연결해볼 것: threshold 조정, permutation importance, SHAP.
