import re
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

import support

if __name__ == '__main__':
    # list of benchmark algorithms
    models = [
        ('SVM', SVC(random_state=1), SVR()),
        ('GaussianProcess', GaussianProcessClassifier(random_state=1),
         GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1)),
        ('KNeighbors', KNeighborsClassifier(), KNeighborsRegressor()),
        ('MLP', MLPClassifier(random_state=1), MLPRegressor(hidden_layer_sizes=(5), solver='lbfgs',
                                                            random_state=1)),
    ]

    # validation dataset list and list of separator and location of header and index
    classifier_files = ['./data/iris.data',
                        './data/sonar.all-data', './data/glass.data']
    classifier_params = [(',', None, None), (',', None, None), (',', None, 0)]
    regressor_files = ['./data/airfoil_self_noise.dat',
                       './data/winequality-red.csv', './data/winequality-white.csv']
    regressor_params = [('\t', None, None), (';', 0, None), (';', 0, None)]

    # result table
    result = pd.DataFrame(columns=['target', 'function'] + [m[0] for m in models],
                          index=range(len(classifier_files+regressor_files)*2))

# evaluate classifier algorithms at first.
ncol = 0
for i, (c, p) in enumerate(zip(classifier_files, classifier_params)):
    df = pd.read_csv(c, sep=p[0], header=p[1], index_col=p[2])
    x = df[df.columns[:-1]].values
    y, clz = support.clz_to_prob(df[df.columns[-1]])

    result.loc[ncol, 'target'] = re.split(r'[._]', c)[0]
    result.loc[ncol+1, 'target'] = ''
    result.loc[ncol, 'function'] = 'F1Score'
    result.loc[ncol+1, 'function'] = 'Accuracy'

    # implement evaluating algorithm

    for l, c_m, r_m in models:
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        s = cross_validate(c_m, x, y.argmax(axis=1), cv=kf,
                           scoring=('f1_weighted', 'accuracy'))
        result.loc[ncol, l] = np.mean(s['test_f1_weighted'])
        result.loc[ncol+1, l] = np.mean(s['test_accuracy'])

    ncol += 2

# Next, evaluate regression algorithms
for i, (c, p) in enumerate(zip(regressor_files, regressor_params)):
    df = pd.read_csv(c, sep=p[0], header=p[1], index_col=p[2])
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values.reshape(-1,)
    result.loc[ncol, 'target'] = re.split(r'[._]', c)[0]
    result.loc[ncol+1, 'target'] = ''
    result.loc[ncol, 'function'] = 'R2Score'
    result.loc[ncol+1, 'function'] = 'MeanSquared'

    for l, c_m, r_m in models:
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        s = cross_validate(r_m, x, y, cv=kf, scoring=(
            'r2', 'neg_mean_squared_error'))
        result.loc[ncol, l] = np.mean(s['test_r2'])
        result.loc[ncol+1, l] = -np.mean(s['test_neg_mean_squared_error'])

    ncol += 2

# save result
print(result)
result.to_csv('baseline.csv', index=None)
