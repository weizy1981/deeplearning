from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# 导入数据
dataset = datasets.load_boston()

x = dataset.data
Y = dataset.target

# 设定随机种子
seed = 7
np.random.seed(seed)

# 构建模型函数
def create_model(units_list=[13],optimizer='adam', init='normal'):
    # 构建模型
    model = Sequential()

    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))
    # 构建更多隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))

    model.add(Dense(units=1, kernel_initializer=init))

    # 编译模型
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=0)


# 设置算法评估基准
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))

# 数据正态化，改进算法
steps = []
steps.append(('standardize', StandardScaler()))
steps.append(('mlp', model))
pipeline = Pipeline(steps)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x, Y, cv=kfold)
print('Standardize: %.2f (%.2f) MSE' % (results.mean(), results.std()))

# 调参选择最优模型
param_grid = {}
param_grid['units_list'] = [[20], [13, 6]]
param_grid['optimizer'] = ['rmsprop', 'adam']
param_grid['init'] = ['glorot_uniform', 'normal']
param_grid['epochs'] = [100, 200]
param_grid['batch_size'] = [5, 20]

# 调参
scaler = StandardScaler()
scaler_x = scaler.fit_transform(x)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(scaler_x, Y)

# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))
