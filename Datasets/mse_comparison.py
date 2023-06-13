import Trees
from StackedSingleTarget import RegressorStackedSingleTarget, ClassifierStackedSingleTarget
from Chains import RegressorChain, ClassifierChain
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.io.arff import loadarff
from sklearn.metrics import hamming_loss
import datetime
import statistics
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
import time
import rf
import gb


def load_data(path=None, n_features=None, df=None):
    if df is None:
        raw_data = loadarff(path)
        df = pd.DataFrame(raw_data[0])
    X = df.iloc[:, range(n_features)].to_numpy()
    y = df.iloc[:, range(n_features, df.shape[1])].to_numpy()
    return X, y


def predict_cut_only_for_all(X_train, X_test, Y_train, Y_test, max_depth=None, min_samples_leaf=None, use_pruning=True,
                             print_tree=True, model='decision tree', regression=True, n_estimators=None,
                             learning_rate=0.1):
    start_time = time.time()
    if model == "decision tree":
        if regression:
            regressor_only_for_all = Trees.CartDecisionTreeRegressor(max_depth=max_depth, use_pruning=use_pruning)
        else:
            regressor_only_for_all = Trees.CartDecisionTreeClassifier(max_depth=max_depth, use_pruning=use_pruning)
    elif model == "gb":
        if regression:
            regressor_only_for_all = gb.GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                                                  learning_rate=learning_rate,
                                                                  min_samples_leaf=min_samples_leaf)
        else:
            regressor_only_for_all = gb.GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                                  learning_rate=learning_rate,
                                                                  min_samples_leaf=min_samples_leaf)

    else:
        raise ValueError(f'model parameter get only "decision tree","rf" or "gb" but got {model} instead.')
    regressor_only_for_all.fit(X_train, Y_train)
    result_only_for_all = regressor_only_for_all.predict(X_test)
    if print_tree:
        if model == "decision tree":
            regressor_only_for_all.print_tree()
        # else:
        # regressor_only_for_all.print_forest()
    print(f'runtime for all model:{time.time() - start_time} seconds')
    del regressor_only_for_all
    return result_only_for_all


def predict_cut_only_for_individual(X_train, X_test, Y_train, Y_test, max_depth=None, min_samples_leaf=None,
                                    use_pruning=True,
                                    print_tree=True, model='decision tree', regression=True, n_estimators=None, learning_rate=0.1):
    start_time = time.time()
    result_only_for_individual = list()
    for i in range(Y_train.shape[1]):
        y_i = Y_train[:, [i]]  # Take only y_i
        if model == "decision tree":
            if regression:
                regressor_only_for_individual = Trees.CartDecisionTreeRegressor(max_depth=max_depth,
                                                                                use_pruning=use_pruning)
            else:
                regressor_only_for_individual = Trees.CartDecisionTreeClassifier(max_depth=max_depth,
                                                                                 use_pruning=use_pruning)
        elif model == 'rf':
            if regression:
                regressor_only_for_individual = rf.RandomForestRegrresor(max_depth=max_depth, n_estimators=n_estimators,
                                                                         min_samples_leaf=min_samples_leaf)
            else:
                regressor_only_for_individual = rf.RandomForestClassifier(max_depth=max_depth,
                                                                          n_estimators=n_estimators,
                                                                          min_samples_leaf=min_samples_leaf)
        elif model == "gb":
            if regression:
                regressor_only_for_individual = gb.GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                                                      learning_rate=learning_rate,
                                                                      min_samples_leaf=min_samples_leaf)
            else:
                regressor_only_for_individual = gb.GradientBoostingClassifier(max_depth=max_depth,
                                                                             n_estimators=n_estimators,
                                                                             learning_rate=learning_rate,
                                                                             min_samples_leaf=min_samples_leaf)


        else:
            raise ValueError(f'model parameter get only "decision tree","rf" or "gb" but got {model} instead.')
        regressor_only_for_individual.fit(X_train, y_i)
        predict_yi = regressor_only_for_individual.predict(X_test)
        predict_yi = [item for sublist in predict_yi for item in sublist]  # create flat list
        result_only_for_individual.append(predict_yi)
        if print_tree:
            print(f'print for target number: {i}')
            if model == "decision tree":
                regressor_only_for_individual.print_tree()
        # else:
        #  regressor_only_for_individual.print_forest()
    result_only_for_individual = np.array(result_only_for_individual)
    result_only_for_individual = np.transpose(result_only_for_individual)
    print(f'runtime for ind model:{time.time() - start_time} seconds')
    return result_only_for_individual


def predict_LOO(X_train, X_test, Y_train, Y_test, max_depth=np.inf, min_samples_leaf=None, use_pruning=False,
                cv=KFold(n_splits=10, shuffle=True, random_state=1),
                model='decision tree', regression=True,
                n_estimators=None,learning_rate=0.1):  # High max depth- used to stop when score decrease
    start_time = time.time()
    if model == "decision tree":
        if regression:
            regressor_LOO = Trees.LooDecisionTreeRegressor(max_depth=max_depth, use_pruning=use_pruning, cv=cv)
        else:
            regressor_LOO = Trees.LooDecisionTreeClassifier(max_depth=max_depth, use_pruning=use_pruning, cv=cv)
    elif model == "rf":
        if regression:
            regressor_LOO = rf.LooRandomForestRegrresor(max_depth=max_depth, n_estimators=n_estimators,
                                                        min_samples_leaf=min_samples_leaf)
        else:
            regressor_LOO = rf.RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                      min_samples_leaf=min_samples_leaf)
    elif model == "gb":
        if regression:
            regressor_LOO = gb.KfoldGradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                                                     learning_rate=learning_rate,
                                                                     min_samples_leaf=min_samples_leaf)
        else:
            regressor_LOO = gb.KfoldGradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                              learning_rate=learning_rate,
                                                              min_samples_leaf=min_samples_leaf)

    else:
        raise ValueError(f'model parameter get only "decision tree","rf" or "gb" but got {model} instead.')

    regressor_LOO.fit(X_train, Y_train)
    if model == "decision tree":
        regressor_LOO.print_tree()
    # else:
    #  regressor_LOO.print_forest()
    result_LOO = regressor_LOO.predict(X_test)
    print(f'runtime for LOO model:{time.time() - start_time} seconds')
    return result_LOO


def predict_choose_one_model(X_train, X_test, Y_train, Y_test, use_pruning=True, regression=True, model='decision tree',
                             cv=KFold(n_splits=10, shuffle=True, random_state=0), print_tree=False, n_estimators=None,min_samples_leaf=1,learning_rate=0.1):
    start_time = time.time()
    print('starting choose one model')
    mse_ind, mse_all = 0, 0
    for train_index, test_index in tqdm(cv.split(X_train), desc='Choose one progress bar'):
        x_train_cv, x_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = Y_train[train_index], Y_train[test_index]
        print('Starting all')
        res_all = predict_cut_only_for_all(x_train_cv, x_test_cv, y_train_cv, y_test_cv, use_pruning=use_pruning,
                                           model=model, regression=regression, print_tree=print_tree,n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,
                                           learning_rate=learning_rate)
        print('Finish all')
        print('Starting ind')
        res_ind = predict_cut_only_for_individual(x_train_cv, x_test_cv, y_train_cv, y_test_cv, use_pruning=use_pruning,
                                                  model=model, regression=regression, print_tree=print_tree,n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,
                                                  learning_rate=learning_rate)
        print('Finish ind')
        mse_all += mean_squared_error(y_test_cv, res_all)
        # mse_ind += mean_squared_error(y_test_cv, res_ind)

    print('MSE all:', mse_all)
    print('MSE ind:', mse_ind)

    if mse_all <= mse_ind:
        print('for all was chosen')
        res = predict_cut_only_for_all(X_train, X_test, Y_train, Y_test, model=model, regression=regression,
                                       n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,
                                       print_tree=print_tree,learning_rate=learning_rate)
        print(f'runtime for choose one model:{time.time() - start_time} seconds')
        return res

    else:
        print('for ind was chosen')
        res = predict_cut_only_for_individual(X_train, X_test, Y_train, Y_test, model=model, regression=regression,
                                              n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,
                                              print_tree=print_tree,learning_rate=learning_rate)
        print(f'runtime for choose one model:{time.time() - start_time} seconds')
        return res


def predict_all_models(X_train, X_test, Y_train, Y_test, max_depth=None, min_samples_leaf=None,
                       use_pruning_for_loo=False, use_pruning_2_models=False,
                       model='decision tree', regression=True, n_estimators=None,learning_rate=0.1):
     print('for all print:')
     pred_all = predict_cut_only_for_all(X_train, X_test, Y_train, Y_test, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          use_pruning=use_pruning_2_models, model=model, regression=regression,
                                         n_estimators=n_estimators,learning_rate=learning_rate)
     print('for ind print:')
     pred_ind = predict_cut_only_for_individual(X_train, X_test, Y_train, Y_test, max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 use_pruning=use_pruning_2_models, model=model, regression=regression,
                                                 n_estimators=n_estimators,learning_rate=learning_rate)
     print('LOO print:')
     pred_loo = predict_LOO(X_train, X_test, Y_train, Y_test,
                           use_pruning=use_pruning_for_loo, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                           model=model, regression=regression, n_estimators=n_estimators,learning_rate=learning_rate)
    # pred_choose = predict_choose_one_model(X_train, X_test, Y_train, Y_test,use_pruning=use_pruning_2_models)
     return pred_all, pred_ind, pred_loo, None


def mse_comparison(path, n_features, max_depth=None, min_samples_leaf=None, df=None, use_pruning_for_loo=False,
                   use_pruning_2_models=False, only_choose_one=False, n_splits=5, only_related_works = False,
                   n_repeats=1, random_state=42, model='decision tree', regression=True, n_estimators=None,learning_rate=0.1):
    print(f'Cell execution time: {datetime.datetime.now()}')
    start = time.time()
    X, y = load_data(path=path, df=df, n_features=n_features)
    mse_all, mse_ind, mse_loo, mse_choose = [], [], [], []
    mse_chain, mse_stacked = [], []
    for repeat in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for train_index, test_index in tqdm(cv.split(X), desc='CV progress bar'):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if only_choose_one:
                pred_choose = predict_choose_one_model(X_train, X_test, y_train, y_test, use_pruning=True,
                                                       regression=regression, model=model,n_estimators=n_estimators,
                                                       min_samples_leaf=min_samples_leaf,learning_rate=learning_rate)
                if regression:
                    scoring_choose = np.mean(mean_squared_error(y_test, pred_choose))
                else:
                    scoring_choose = hamming_loss(y_test, pred_choose)
                mse_choose.append(scoring_choose)
            elif only_related_works:
                if regression:
                    chain = RegressorChain()
                    stacked = RegressorStackedSingleTarget()
                else:
                    chain = ClassifierChain()
                    stacked = ClassifierStackedSingleTarget()
                chain.fit(X_train, y_train)
                pred_chain = chain.predict(X_test)
                stacked.fit(X_train, y_train)
                pred_stacked = stacked.predict(X_test)
                if regression:
                    score_chain = np.mean(mean_squared_error(y_test, pred_chain))
                    score_stacked = np.mean(mean_squared_error(y_test, pred_stacked))
                else:
                    score_chain = hamming_loss(y_test, pred_chain)
                    score_stacked = hamming_loss(y_test, pred_stacked)
                mse_chain.append(score_chain)
                mse_stacked.append(score_stacked)
                print('List of MSE chains:', mse_chain)
                print('List of MSE stacked:', mse_stacked)
            else:

                pred_all, pred_ind, pred_loo, pred_choose = predict_all_models(X_train, X_test, y_train, y_test,
                                                                               max_depth=max_depth,
                                                                               min_samples_leaf=min_samples_leaf,
                                                                               use_pruning_for_loo=use_pruning_for_loo,
                                                                               use_pruning_2_models=use_pruning_2_models,
                                                                               model=model, regression=regression,
                                                                               n_estimators=n_estimators,learning_rate=learning_rate)
                if regression:
                    scoring_all = np.mean(mean_squared_error(y_test, pred_all, multioutput='raw_values'))
                    scoring_ind = np.mean(mean_squared_error(y_test, pred_ind, multioutput='raw_values'))
                    scoring_loo = np.mean(mean_squared_error(y_test, pred_loo, multioutput='raw_values'))
                else:
                    scoring_all = hamming_loss(y_test, pred_all)
                    scoring_ind = hamming_loss(y_test, pred_ind)
                    scoring_loo = hamming_loss(y_test, pred_loo)

                mse_all.append(scoring_all)
                mse_ind.append(scoring_ind)
                mse_loo.append(scoring_loo)
                print(f'Score for all:{scoring_all}')
                print(f'Score for ind:{scoring_ind}')
                print(f'Score for loo model:{scoring_loo}')

                print('****' * 40)
    print("---" * 30)
    print('MSE chains :', np.array(mse_chain).mean())
    print('MSE stacked:', np.array(mse_stacked).mean())
    print('stdev choose chains model', statistics.stdev(mse_chain))
    print('stdev choose stacked model', statistics.stdev(mse_stacked))

    print(f'MSE for all model:   {np.array(mse_all).mean()}')
    print(f'MSE for ind model:   {np.array(mse_ind).mean()}')
    print(f'MSE LOO model:       {np.array(mse_loo).mean()}')
    if len(mse_choose) > 0:
        print(f'MSE choose one model:{np.array(mse_choose).mean()}')
        print(f'stdev choose one model:{statistics.stdev(mse_choose)}')

    print(f'Total runtime: {(time.time() - start) / 60} minutes')

    return mse_all, mse_ind, mse_loo, mse_choose


def statistical_analysis(path,df,n_features, model,regression = True, random_state=42):
    print(f'Cell execution time: {datetime.datetime.now()}')
    start = time.time()
    X, y = load_data(path=path, df=df, n_features=n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_state)
    min_samples_leaf = int(0.05 * len(X_train))
    pred_all = predict_cut_only_for_all(X_train, X_test, y_train, y_test, model=model, regression=regression,min_samples_leaf=min_samples_leaf,n_estimators=50)
    pred_ind = predict_cut_only_for_individual(X_train, X_test, y_train, y_test,model=model,regression=regression,min_samples_leaf=min_samples_leaf,n_estimators=5)
    pred_loo = predict_LOO(X_train, X_test, y_train, y_test,model=model,regression=regression,min_samples_leaf=min_samples_leaf,n_estimators=50)
    #pred_choose = predict_choose_one_model(X_train, X_test, y_train, y_test,model=model, regression=regression,min_samples_leaf=min_samples_leaf,n_estimators=50)

    if regression:
        scoring_all = np.mean(np.subtract(y_test, pred_all) ** 2,axis=1)
        scoring_ind = np.mean(np.subtract(y_test, pred_ind) ** 2,axis=1)
        scoring_loo = np.mean(np.subtract(y_test, pred_loo) ** 2,axis=1)
    else:
        scoring_all = np.mean((y_test !=  np.array(pred_all)).astype(int),axis=1)
        scoring_ind = np.mean((y_test !=  np.array(pred_ind)).astype(int),axis=1)
        scoring_loo = np.mean((y_test !=  np.array(pred_loo)).astype(int),axis=1)

    t_test_for_all = stats.ttest_rel(scoring_loo, scoring_all, alternative='less')
    t_test_for_ind = stats.ttest_rel(scoring_loo, scoring_ind, alternative='less')
    #t_test_for_choose = stats.ttest_rel(pred_loo, pred_choose, alternative='less')

    print("The alternative hypothesis is that the mean predictions of our method is less than MT/ST")
    print(10*"****")

    print(f'The resuls of the t-test on our method and MT are: {t_test_for_all}')
    print(f'The resuls of the t-test on our method and ST are: {t_test_for_ind}')
   # print(f'The resuls of the t-test on our method and MT are: {t_test_for_choose}')

    print(f'Results for MT method: {scoring_all.mean()}')
    print(f'Results for ST method: {scoring_ind.mean()}')
    print(f'Results for our method: {scoring_loo.mean()}')

    print(f'Total runtime: {(time.time() - start) / 60} minutes')







def scale_df(df, n_targets, scale_method):
    df_1 = df.copy()
    y = df_1.iloc[:, -n_targets:]
    if scale_method == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.StandardScaler()
    y_norm = scaler.fit_transform(y)
    df_1.iloc[:, -n_targets:] = y_norm
    return df_1

if __name__ == "__main__":
    df = pd.read_csv('Pakistan_Largest_Ecommerce_Dataset.csv')


    def move_columns_to_end(df, col1, col2=None):
        cols = list(df.columns.values)  # Make a list of all of the columns in the df
        cols.pop(cols.index(col1))  # Remove col1 from list
        if col2:
            cols.pop(cols.index(col2))  # Remove col2 from list
        if col2:
            df = df[cols + [col1, col2]]  # Create new dataframe with columns in the order you want
        else:
            df = df[cols + [col1]]
        return df


    def preprocessing(df):
        df1 = df.copy()
        df1['Mobiles & Tablets'] = df['Mobiles & Tablets'].apply(
            lambda x: 0 if x <= df['Mobiles & Tablets'].median() else 1)
        df1['Beauty & Grooming'] = df['Beauty & Grooming'].apply(
            lambda x: 0 if x <= df['Beauty & Grooming'].median() else 1)
        return df1


    pakistan_columns = ['grand_total', 'category_name_1', 'payment_method', 'Customer ID']
    pakistan_filtered = df[pakistan_columns]
    grouped_data = pakistan_filtered.groupby(by=['category_name_1', 'Customer ID'], as_index=False).agg(
        {'grand_total': 'sum'})
    pivoted = grouped_data.pivot(index='Customer ID', columns='category_name_1', values='grand_total')
    pivoted = pivoted.reset_index().rename_axis(None, axis=1)

    final_df = pivoted[pivoted.notnull().sum(axis=1) >= 9]
    final_df = final_df.drop(['Customer ID'], axis='columns')
    final_df = final_df.fillna(0)
    print(final_df.shape)

    final_df = move_columns_to_end(final_df, 'Mobiles & Tablets', 'Beauty & Grooming')
    final_df_classifiction = preprocessing(final_df)

    statistical_analysis(path=None,df=final_df_classifiction, model='gb', n_features=14)
    #
    # res_Kfold_all, res_Kfold_ind, res_Kfold_loo, res_Kfold_choose = mse_comparison(path=None,
    #                                                                                               df=final_df_classifiction,
    #                                                                                               n_features=14,
    #                                                                                               max_depth=np.inf,
    #                                                                                               min_samples_leaf=int(
    #                                                                                                   0.05 *
    #                                                                                                   final_df_classifiction.shape[
    #                                                                                                       0]),
    #                                                                                               n_splits=10,
    #                                                                                               model='gb',
    #                                                                                               n_estimators=50,
    #                                                                                               regression=False,
    #                                                                                               random_state=0)
