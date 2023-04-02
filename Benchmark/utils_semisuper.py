import numpy as np
import pandas as pd
from scipy.stats import genextreme

from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
from statsmodels.discrete.discrete_model import MNLogit
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# Параметры, использующиеся в статье:
    
# n = 1000
# rho = np.array([[1.00,  0.10,  0.45,  0.64 * np.sqrt(32.00)],
#                 [0.10,  1.00, -0.35, -0.24 * np.sqrt(32.00)],
#                 [0.45, -0.35,  1.00,  0.14 * np.sqrt(32.00)],
#                 [0.64 * np.sqrt(32.00), -0.24 * np.sqrt(32.00),  0.14 * np.sqrt(32.00), 32.00]])

# r = 0.9
# betas = np.array([1, 1, 1])
# gammas = np.array([[3.3, 0.0, 0.0], # в оригинале 0.9, но мб надо 3.3
#                    [1.0, 1.0, 0.0],
#                    [1.0, 0.0, 1.0]])

def get_benchmark_data(n, rho, r, betas, gammas):
    '''
    Генерирует данные на основе (Bourguignon et al, 2007)
    
    Параметры:
        n - размер всей выборки (таргет в основном уравнении будет виден лишь у около трети наблюдений)
        rho - ковариационная матрица многомерного нормального распределения случайных ошибок
        r - параметр, использующийся для генерации регрессоров основного уравнения
        betas - коэффициенты основного уравнения
        gammas - коэффициенты уравнений отбора
        
        
    Примечание: для z != 1 значения y не наблюдаются
        
    '''
    
    errors = np.random.multivariate_normal(mean=np.zeros(4), cov=rho, size=n)
    us, eps = errors[:, :-1], errors[:, -1]
    vs = np.random.normal(size=(n, 2), loc=0, scale=(4 * np.sqrt(1 - r ** 2)))
    
    W = np.random.normal(size=(n, 2), loc=0, scale=4)
    X = r * W + vs
    W_w_const = np.hstack((np.ones((n, 1)), W))
    X_w_const = np.hstack((np.ones((n, 1)), X))
    
    Z_star = W_w_const @ gammas + us
    z = np.argmax(Z_star, axis=1) + 1
    y = X_w_const @ betas + eps
    
    data = np.hstack((W_w_const, X_w_const, Z_star, 
                      z.reshape((n, 1)), y.reshape((n, 1))))
    cols = ['w0', 'w1', 'w2','x0', 'x1', 'x2',
               'z_star1', 'z_star2', 'z_star3', 'z', 'y']
    df = pd.DataFrame(data, columns=cols)
    df['y_star'] = df['y']
    df.loc[df.z != 1, 'y'] = np.nan
    
    return df


def calc_metrics(ests, true, model_name):
    return [model_name] + \
           list(np.mean(ests, axis=0)) + \
           list(np.var(ests, axis=0)) + \
           list(100 * np.mean(np.abs(np.array(ests) - true) / true, axis=0))


def run_simulations(n, rho, r, betas, gammas, n_simulations=1000):
    
    ols_results = []
    dmf_results = []
    ml_results = []
    semisuper_results = []
    
    for i in tqdm(range(n_simulations)):
        
        df = get_benchmark_data(n, rho, r, betas, gammas)
        df_no_nans = df.dropna()
        
        W = df[['w0', 'w1', 'w2']]
        z = df['z']
        
        X = df[['x1', 'x2']]
        y = df['y']
        
        X_no_nans = df_no_nans[['x1', 'x2']]
        y_no_nans = df_no_nans['y']
    
        ## OLS
        ols = LinearRegression().fit(X_no_nans, y_no_nans)
        ols_results.append([ols.intercept_] + list(ols.coef_))
        
        ## DMF
        lr = MNLogit(z, W).fit(disp=0)
        prob_lr = lr.predict()
        
        # Лямбды
        df['lambda1'] = -np.log(prob_lr[:, 0])
        df['lambda2'] = prob_lr[:, 1] * np.log(prob_lr[:, 1]) / (1 - prob_lr[:, 1])
        df['lambda3'] = prob_lr[:, 2] * np.log(prob_lr[:, 2]) / (1 - prob_lr[:, 2])
        df_no_nans = df.dropna()
        X_no_nans = df_no_nans[['x1', 'x2', 'lambda1', 'lambda2', 'lambda3']]
        y_no_nans = df_no_nans['y']
        
        # Линейная модель
        dmf = LinearRegression().fit(X_no_nans, y_no_nans)
        dmf_results.append([dmf.intercept_] + list(dmf.coef_[:2]))
        
        ## Бустинг с полиномами
        # Вероятности
        boosting = CatBoostClassifier(iterations=10, max_depth=3, verbose=0)
        boosting.fit(W, z)
        prob_boost = boosting.predict_proba(W)
        
        # Полиномы вероятностей
        # Подбираем лучший по CV с 4 фолдами (по MSE)
        ks = np.arange(1, 7)
        all_mses = []
        for k in ks:
            # генерируем степени вероятностей
            for i in range(1, k+1):
                df[f'proba0^{i}'] = prob_boost[:, 0] ** i
                df[f'proba1^{i}'] = prob_boost[:, 1] ** i
                df[f'proba2^{i}'] = prob_boost[:, 2] ** i
            needed_columns = ['x1', 'x2'] \
                           + [f'proba{category}^{power}' for category in [0, 1, 2] 
                                                         for power in range(1, i)]
            X, y = df.dropna()[needed_columns], df.dropna()['y']
            
            # считаем RMSE по KFold
            mse_for_k = []
            kf = KFold(n_splits=4)
            for train, test in kf.split(X, y):
                X_train, y_train = X.iloc[train], y.iloc[train]
                X_test, y_test = X.iloc[test], y.iloc[test]
                
                lm = LinearRegression().fit(X_train, y_train)
                mse_for_k.append(np.mean((y_test - lm.predict(X_test)) ** 2))
    
            all_mses.append(np.mean(mse_for_k))
            
        best_k = ks[np.argmin(all_mses)]
        
        df_no_nans = df.dropna()
        
        ## Линейная модель
        needed_columns = ['x1', 'x2'] \
                       + [f'proba{category}^{power}' for category in [0, 1, 2] for power in range(1, best_k+1)]
        
        dmf_ml = LinearRegression().fit(df_no_nans[needed_columns], y_no_nans)
        ml_results.append([dmf_ml.intercept_] + list(dmf_ml.coef_[:2]))
        
        
        ## Подход с semi-super
        semisuper_results.append(estimate_semisupervised(df))
    
    ols_res_metrics = calc_metrics(ols_results, betas, 'OLS')
    dmf_res_metrics = calc_metrics(dmf_results, betas, 'DMF')
    ml_res_metrics = calc_metrics(ml_results, betas, 'Catboost')
    semisuper_metrics = calc_metrics(semisuper_results, betas, 'SemiSuper')
    
    cols = ['method', 'beta0_mean', 'beta1_mean', 'beta2_mean', 
            'beta0_sd', 'beta1_sd', 'beta2_sd', 'beta0_MAPE', 'beta1_MAPE', 'beta2_MAPE']
    
    cols_order_show = ['method', 'beta0_mean', 'beta0_sd', 'beta0_MAPE', 
                       'beta1_mean', 'beta1_sd', 'beta1_MAPE',
                       'beta2_mean', 'beta2_sd', 'beta2_MAPE']
    
    results = pd.DataFrame([ols_res_metrics, 
                            dmf_res_metrics,
                            ml_res_metrics,
                            semisuper_metrics], columns=cols)
    
    return results[cols_order_show]


def estimate_semisupervised(df):
    # признаки для восстановления пропущенных значений
    recoveries = ['w1', 'w2', 'x1', 'x2', 'z']
    
    X_train = df.loc[~df.y.isna(), recoveries]
    y_train = df.loc[~df.y.isna(), 'y']
    
    X_test = df.loc[df.y.isna(), recoveries]
    y_test = df.loc[df.y.isna(), 'y_star']
    
    # подход
    param_grid = {'C': list(0.1 ** np.arange(5)) + list(np.arange(2, 11)), 
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    
    svm = SVR(max_iter=1000000)
    
    searcher = GridSearchCV(svm, param_grid, scoring='neg_root_mean_squared_error', n_jobs=5, cv=5, refit=True)
    searcher.fit(X_train, y_train)
    
    y_pred = searcher.best_estimator_.predict(X_test)
    df['y_pred'] = df['y']
    df.loc[df.y.isna(), 'y_pred'] = y_pred
    X = df[['x1', 'x2']]
    y = df['y_pred']
    
    # Финальный МНК
    ols = LinearRegression().fit(X, y)
    return [ols.intercept_] + list(ols.coef_)


        




