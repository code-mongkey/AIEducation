import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.linear_model import Lasso, LassoCV

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# : , 본문 내용 능형 라쏘
# : ISL 데이터 정의 데이터
infile = "data/Credit.csv"
data = pd.read_csv(infile)
data.head()
data.shape
y = data['Balance'].copy()
X = data.copy()
X.drop(['Balance'], axis=1, inplace=True)
X.Gender = X.Gender.map({'Female': 1, ' Male': 0})
X.Student = X.Student.map({'Yes': 1, 'No': 0})
X.Married = X.Married.map({'Yes': 1, 'No': 0})
X = X.join(pd.get_dummies(X['Ethnicity']))
X.drop(['Ethnicity'], axis=1, inplace=True)
features = X[['Asian', 'Caucasian', 'African American']]
predictors = X.columns
# 표준화
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))
# : Ridge 본문 내용 생성
lambdas = np.logspace(-2, 6, 100)
coefficients = np.zeros(shape=(12, len(lambdas)))
for i, l in enumerate(lambdas):
 ridgeReg = Ridge(alpha=l)
 ridgeReg.fit(X, y)
 coefficients[:, i] = ridgeReg.coef_
plt.figure()
plt.title("LASSO 정규화")
plt.xlabel("$\lambda$")
plt.ylabel("표쥰화 회귀계수")
styles = ['-', '--', '-.', ':']
xticks = [0.01, 1, 100, 1e4]
labels = ['1e-02', '1e+00', '1e+02', '1e+04']
plt.gca().set_xscale("log", basex=10)
plt.gca().set_xticks(xticks)
plt.gca().set_xticks([], minor=True)
plt.gca().set_xticklabels(labels)
for i in range(0, 12):
    s = styles[i % len(styles)]
    if i < 3 or i == 7:
        plt.plot(lambdas, coefficients[i], label=predictors[i], linestyle=s)
    else:
        plt.plot(lambdas, coefficients[i], color='lightgray')
plt.legend(loc='best')
plt.savefig(png_path + "/regularization_credit_ridge.png")
# : Lasso 본문 내용 생성
lambdas = np.logspace(-2, 1, 100)
coefficients = np.zeros(shape=(12, len(lambdas)))
for i, l in enumerate(lambdas):
    lassoReg = Lasso(alpha=l)
    lassoReg.fit(X, y)
    coefficients[:, i] = lassoReg.coef_
plt.figure()
# plt.title("Lasso Regularization")
plt.title("LASSO 정규화")
plt.xlabel("$\lambda$")
plt.ylabel("표준화 회귀계수")
styles = ['-', '--', '-.', ':']
xticks = [0.01, 0.1, 0.5, 1, 10]
labels = [0.01, 0.1, 0.5, 1, 10]
# labels = ['1e-02', '1e+00', '1e+02', '1e+04']
# xticks = [1, 10, 20, 50, 100, 200, 500, 1000, 5000]
# labels = ['1', '10', '20', '50', '100', '200', '500', '1000', '5000']
plt.gca().set_xscale("log", basex=10)
plt.gca().set_xticks(xticks)
plt.gca().set_xticks([], minor=True)
plt.gca().set_xticklabels(labels)
for i in range(0, 12):
    s = styles[i % len(styles)]
    if i < 3 or i == 7:
        plt.plot(lambdas, coefficients[i], label=predictors[i], linestyle=s)
    else:
        plt.plot(lambdas, coefficients[i], color='lightgray')
plt.legend(loc='best')
plt.savefig(png_path + "/regularization_credit_lasso.png")
# : LASSO vs. Ridge 본문 내용 그래프 비교 벌점함수 : 영역과 비용 함수
from matplotlib.patches import Ellipse, Circle, RegularPolygon
fig = plt.figure(figsize=(10, 5), facecolor='w')
cx, cy = (5.9, 11.1)
ax = fig.add_subplot(121, frameon=True, xticks=[], yticks=[])
ax.arrow(-6, 0, 18, 0, head_width=0.5, fc='k', lw=1.5)
ax.arrow(0, -6, 0, 18, head_width=0.5, fc='k', lw=1.5)

for i in [0, 0.6, 2]:
    ax.add_patch(Ellipse((cx, cy),
    1.1 * np.sqrt(11 * i + 5), 3.5 * np.sqrt(11 * i + 5),
    -45, fc='none', color='r', lw=1.5))

ax.add_patch(RegularPolygon((0, 0), 4, 4.4, np.pi, fc='gray'))
ax.scatter(cx, cy, marker='o', s=10, color='k')
ax.text(11.5, -0.5, r'$\beta_1$', va='top', fontsize=13)
ax.text(-0.5, 11.5, r'$\beta_2$', ha='right', fontsize=13)
ax.text(cx - 0.6, cy, r'$\rm \hat{\beta}$', ha='center', va='center', fontsize=13)
ax.set_xlim(-6, 20)
ax.set_ylim(-6, 20)
plt.axis('off')
cx, cy = (6.9, 11.04)
ax = fig.add_subplot(122, frameon=True, xticks=[], yticks=[])
ax.arrow(-6, 0, 18, 0, head_width=0.5, fc='k', lw=1.5)
ax.arrow(0, -6, 0, 18, head_width=0.5, fc='k', lw=1.5)

for i in [0, 0.6, 2]:
    ax.add_patch(Ellipse((cx, cy),
    1.1 * np.sqrt(11 * i + 5), 3.5 * np.sqrt(11 * i + 5),
    -45, fc='none', color='r', lw=1.5))

# ax.add_patch(Circle((0, 0), 4.4, fc='#6BC6C1'))
ax.add_patch(Circle((0, 0), 4.4, fc='gray'))
ax.scatter(cx, cy, marker='o', s=10, color='k')
ax.text(11.5, -0.5, r'$\beta_1$', va='top', fontsize=13)
ax.text(-0.5, 11.5, r'$\beta_2$', ha='right', fontsize=13)
ax.text(cx - 0.6, cy, r'$\rm \hat{\beta}$', ha='center', va='center', fontsize=13)
ax.set_xlim(-6, 20)
ax.set_ylim(-6, 20)
plt.axis('off')
plt.tight_layout()
# plt.suptitle('라쏘 대 능형의 벌점함수 영역과 비용 함수')
plt.show()
plt.savefig(png_path + "/regularization_ridge_lasso_comparison.png")
# : [BANK] 예제 데이터에 대한 능형 라쏘 , 적합
np.random.seed(1) # for reproducibility
# 데이터 가져오기
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/'
zip_url = path + 'bank.zip'
z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content)) # 짚 파일 풀기
z.infolist() # 짚 파일 내의 구성 요소 보기
df = pd.read_csv(z.open('bank.csv'),sep=';') # 특정 요소 가져오기
df.columns
# : 로지스틱 함수의 적합 모든 설명 변수가 있는 경우
# 데이터 정의
# get_dummies 가변수 구성을 위한 이해하기
pd.get_dummies([0,1,0,1,2])
pd.get_dummies([0,1,0,1,2], drop_first=True)
# 문자 변수를 숫자 변수로 치환하기
df['num_y'] = pd.get_dummies(df['y'], drop_first=True)
# 범주형 변수명 가져오기
categorical_vars = df.drop(['y', 'num_y'], axis=1).columns[df.drop(['y', 'num_y'],
axis=1).dtypes == 'object']
# 숫자형 변수명 가져오기
num_vars = df.drop(['y', 'num_y'], axis=1).columns[df.drop(['y', 'num_y'],
axis=1).dtypes != 'object']
# 숫자형 변수에 대한 표준화
scaler = preprocessing.StandardScaler()
num_data= pd.DataFrame(scaler.fit_transform(df[num_vars]))
num_data.columns = num_vars
# 범주형 변수에 대한 가변수 구성하기
dumm_data = pd.get_dummies(df[categorical_vars], prefix_sep='_', drop_first=True)
# 가변수와 숫자형 변수만을 이용한 입력 특징 데이터 구성하기
Xdf = num_data.join(dumm_data)[num_vars.tolist() + dumm_data.columns.tolist()]
X = Xdf.values
# 목표 변수 구성하기
y = df['num_y'].values
# , 훈련 평가용 데이터 분할하기
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3 )
# 손실 함수의 정의
def cost (type, alpha, y_true, y_predict, coef, intercept):
    if type == 'Ridge':
        return np.sum((y_predict - y_true)**2) + alpha * (np.sum(coef**2) + intercept**2)
    else:
        return np.sum((y_predict - y_true) ** 2) + alpha * (np.sum(np.abs(coef)) +

abs(intercept))
# 능형 회귀 적합
# ( 추정해야 할 모수 절편 제외)
nparams = train_X.shape[1]
# 람다 값의 범위
lambdas = np.logspace(-2, 6, 100, base=10)
# 모수 값을 넣기 위한 행렬 생성 하나의 : 열이 동일한 람다 값
coefficients = np.zeros(shape=(nparams, len(lambdas)))
# 비용 값을 넣기 위한 벡터 생성
costs = np.zeros(shape=(len(lambdas,)))
# 능형 회귀 적합 및 비용 계산

for i, l in enumerate(lambdas):
    ridge = Ridge(alpha=l, random_state=123)
    ridge.fit(train_X, train_y)
    coefficients[:, i] = ridge.coef_
    predicted = ridge.predict(test_X)
    costs[i] = cost('Ridge', l, test_y, predicted, ridge.coef_, 0)

# 비용을 최소화 하는 인덱스 및 람다 값
min_index = np.argmin(costs)
min_alpha = lambdas[min_index]
sel_coeff = coefficients[:, min_index]
# p 큰 값을 주는 개의 변수를 가져옴 능형 :
num_infl_vars = 7
infl_index = np.argsort(np.abs(coefficients[:, min_index]), )[-num_infl_vars:]
# 표준화된 회귀 계수 그래프
plt.figure()
plt.title("람다 대 회귀 계수")
plt.xlabel("$\lambda$")
plt.ylabel("표준화 회귀계수 값")
styles = ['-', '--', '-.', ':']
xticks = [0.01, 1, 100, 1e4]
labels = ['1e-02', '1e+00', '1e+02', '1e+04']
plt.gca().set_xscale("log", basex=10)
plt.gca().set_xticks(xticks)
plt.gca().set_xticks([], minor=True)
plt.gca().set_xticklabels(labels)

for i in np.arange(Xdf.shape[1]):
    s = styles[i % len(styles)]
    if i in infl_index:
        plt.plot(lambdas, coefficients[i], label=Xdf.columns[i], linestyle=s)
    else:
        plt.plot(lambdas, coefficients[i], color='lightgray')

plt.legend(loc='best')
plt.savefig(png_path + "/regularization_ridge.png")
plt.show()
# lasso 회귀 적합
# ( 추정해야 할 모수 절편 제외)
nparams = train_X.shape[1]
# 람다 값의 범위
lambdas = np.logspace(-4, 0, 100)
# 모수 값을 넣기 위한 행렬 생성 하나의 : 열이 동일한 람다 값
coefficients = np.zeros(shape=(nparams, len(lambdas)))
# 비용 값을 넣기 위한 벡터 생성
costs = np.zeros(shape=(len(lambdas,)))

# lasso 회귀 적합 및 비용 계산
for i, l in enumerate(lambdas):
    lasso = Lasso(alpha=l, random_state=123)
    lasso.fit(train_X, train_y)
    coefficients[:, i] = lasso.coef_
    predicted = lasso.predict(test_X)
    costs[i] = cost('L', l, test_y, predicted, lasso.coef_, 0)

# 비용을 최소화 하는 인덱스 및 람다 값
min_index = np.argmin(costs)
min_alpha = lambdas[min_index]
sel_coeff = coefficients[:, min_index]
# 0 회귀 계수 값이 이 아닌 인덱스
infl_index = np.argwhere(coefficients[:, min_index] != 0)
# 선택 변수 비율
len(infl_index)/len(coefficients[:, min_index])
# 0.7857142857142857
# 표준화된 회귀 계수 및 비용 그래프
plt.figure()
plt.title("람다 대 회귀 계수 (LASSO)") 
plt.xlabel("$\lambda$")
plt.ylabel("표준화 회귀계수 값") 
styles = ['-', '--', '-.', ':']
xticks = [0.0001, 0.005, 0.01, 0.05, 0]
labels = [0.0001, 0.005, 0.01, 0.05, 0]
plt.gca().set_xscale("log", basex=10)
plt.gca().set_xticks(xticks)
plt.gca().set_xticks([], minor=True)
plt.gca().set_xticklabels(labels)

for i in np.arange(Xdf.shape[1]):
    s = styles[i % len(styles)]
    if i in infl_index:
        plt.plot(lambdas, coefficients[i], label=Xdf.columns[i], linestyle=s)
    else:
        plt.plot(lambdas, coefficients[i], color='lightgray')

# plt.legend(loc='best')
plt.savefig(png_path + "/regularization_lasso.png")
plt.show()
# Lasso 모델 평가
# 라쏘 모델에 의하여 선택된 변수
Xdf.columns[infl_index.ravel()]
# 로지스틱 모델 적합
logistic = LogisticRegression(random_state=123)
logistic.fit(train_X[:, infl_index.ravel()], train_y)
y_pred = logistic.predict(test_X[:, infl_index.ravel()])
y_pred_proba = logistic.predict_proba(test_X[:, infl_index.ravel()])[:, 1]
# 평가 데이터에 대한 적합 결과
logistic.score(test_X[:, infl_index.ravel()], test_y)
# ROC 그래프 생성
fpr, tpr, _ = metrics.roc_curve(y_true=test_y, y_score=y_pred_proba)
auc = metrics.roc_auc_score(test_y, y_pred_proba)
# 0.8666174655144652
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=" \n 라쏘 회귀 곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/regularization_lasso_ROC.png')
plt.show()
# ROC 설명을 위한 몇가지 프로그램
(y_pred_proba >= 0.5).sum()
cmat = metrics.confusion_matrix(test_y, y_pred)
# ROC 계산을 위한 데이터 구성
roc_raw = np.sort(np.column_stack((y_pred_proba, y_pred, test_y)), axis=0, )[::-1]
# 0.5 경계값이 인지 확인
# np.sum(roc_raw[:,1] == 1)
# np.sum(roc_raw[:, 0] >= 0.5)
# true positive 인 경우
tp_case = (roc_raw[:, 2] == 1) & (roc_raw[:, 1] == 1)
# false positive 인 경우
fp_case = (roc_raw[:, 2] == 0) & (roc_raw[:, 1] == 1)
np.sum(fp_case)
np.sum(tp_case)
np.sum(test_y)
np.sum(roc_raw[:, 2] == 1)
np.sum(roc_raw[:, 1] == 1)

