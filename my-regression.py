import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# print(train.info(), train.isna().sum())
# print(test.info(), test.isna().sum())
# print(train.head(20))

train_test = [train, test]

# 우선 쓸모없는 정보인 Ticket 삭제
for dataset in train_test:
    dataset.drop("Ticket", axis=1, inplace=True)

# str -> int화 (labeling)
for dataset in train_test:
    dataset["Name"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False) # 추출 부분 기억하기!

# str 범주화
le = LabelEncoder()
# train에는 Embarked 결측치 2개 존재, test에는 결측치 존재 x
c_cols = ['Embarked', 'Sex'] # 라벨링할 수 있는 필드들
for cols in c_cols:
    train[cols] = le.fit_transform(train[cols])
    test[cols] = le.transform(test[cols]) # Name은 train, test가 다른 값이 있어서 transform이 안되는듯 > 직접 매핑하기

# Name은 train, test가 다른 값이 있어서 transform이 안 됨 > 직접 매핑하기
# print(train["Name"].value_counts()) 어떤 이름들이 있나 찾아보기

name_mapping = {
    "Mr" : 0, "Miss" : 1, "Mrs" : 2, "Master" : 3, "Col" : 3,
    "Rev" : 3, "Ms" : 3, "Dr" : 3, "Dona" : 3, "Mme" : 3, "Sir" : 3
}

for dataset in train_test:
    dataset["Name"] = dataset["Name"].map(name_mapping)


for dataset in train_test:
    dataset["family"] = dataset["SibSp"] + dataset["Parch"] + 1

for dataset in train_test:
    dataset.drop(["SibSp", "Parch"], axis=1, inplace=True)

# print(train.info(), test.info()) >> 채워야할 것 Age, Name
train["Name"].fillna(0, inplace=True) # 남자가 많으니 남자로 채우자
# train_1 = train.drop("Cabin", axis=1)
# test_1 = test.drop("Cabin",axis=1)

# 맞춰야 할 값인 Survived 기준으로 상관관계 찾기
# print(train_1.corr()["Survived"].sort_values()) # 상관관계를 보니 이름과 매우 있는 듯, PassengerId와는 매우 관계없음

train.drop(["Fare", "Cabin"], axis=1, inplace=True)
test.drop(["Fare", "Cabin"], axis=1, inplace=True)

# Age 값 채우기
for dataset in train_test:
    dataset["Age"] = dataset["Age"].fillna(dataset.groupby("Name")["Age"].transform("median"))

# print(train.info(), train.head()) 모두 채워졌고 숫자형
# print(test.info()) 모두 채워졌고 숫자형

# 모델 학습
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 학습데이터 입력,정답 분리 & 필요없는거 최종 정리
X = train.drop(["PassengerId", "Survived"], axis=1)
y = train["Survived"]

# 학습데이터에서 학습, 검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 원래 숫자형이었던 데이터들에 대해 scailing도 해주나?
# 라벨로 표현이 가능한 데이터는 scailing 적용 x
# 스케일링을 적용해야 하는 경우: 연속형 데이터: 가격, 나이, 소득 등의 숫자 데이터
# 나이만 한번 적용해보자
numeric_feature = "Age"
# scaler = StandardScaler()

# 연속형 데이터가 거칠 스케일러
# numeric_transformer = Pipeline(
#     steps=[('scaler', StandardScaler())]
# )
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_feature)
#     ], remainder='passthrough' # 나머지 칼럼은 그대로
# )

# 앙상블 모델
ensemble_model = VotingClassifier(
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True))
    ], voting='soft'
)

# 전처리 & 모델링 파이프라인
pipeline = Pipeline(
    steps=[
        # ('preprocessor', preprocessor),
        ('classifier', ensemble_model)
])

# grid search
# 각 dict의 key값들은 이런 특성을 가짐 "Pipline단계이름__모델이름__하이퍼파라미터"
# Pipeline 대문자로 선언한 pipeline의 steps 중 선택하는 것 (preprocessor, classifier)
# ex) classifier__rf__n_estimators 라고하면 pipeline 단계에서 randomforest classifier의 n_estimator를 튜닝한다의 의미!
param_grid = {
    'classifier__rf__n_estimators': [100,200],
    'classifier__rf__max_depth': [0,20]
    # 'classifier__rf__learning_rate': [0.05, 0.1], # random forest에 learning rate이란 파라미터는 없다 뜸 > max_depth 정도 써보자
}
k_fold = KFold(n_splits=10, shuffle=True)

# 최종 적용 
grid_search = GridSearchCV(pipeline, param_grid, cv=k_fold, n_jobs=1, scoring='accuracy')

# 학습
grid_search.fit(X_train, y_train)
print("best param", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 평가
pred = best_model.predict(X_val)
print(f"accuracy score : {accuracy_score(y_val, pred)}")

# 최종 제출하기
test_input = test.drop("PassengerId",axis=1)
idx=test["PassengerId"]

final_pred = best_model.predict(test_input)

submission = pd.DataFrame({
    'PassengerId':idx,
    'Survived':final_pred
}).to_csv("final_submission.csv", index=False)
