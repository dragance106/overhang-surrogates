import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import math
import multiprocessing as mp

from mipt import mipt, mipt_full, LHS_maximin

sample_size = 100   # the number of samples to be selected
repeat_times = 10   # the number of ML models to be trained for each parameter combination,
                    # keeping only their average cvrmse as the final result
num_models = 4      # the number of distinct ML models used in training
                    # at the moment, they are:
                    # #LATER: dnn881, dnn8551, dnn84441, dnn843331, dnn8333331, dnn83332221,
                    # xgb-early5-lr0.3, xgb-early5-lr0.1, xgb-early10-lr0.3, xgb-early10-lr0.1,
                    # #LATER: gbdtpl
num_folds = 5       # number of folds for cross validation

rng = np.random.default_rng()   # random number generator,
                                # used for random splitting in 5-fold cross validation


def train_models(df_training, load):
    """ Uses 5-fold cross validation to train ML models
        Each ML "model" is actually a collection/list of 5 models
        that are trained on different folds of df_training as their test sets
        (with the remaining folds used as training sets).

    :return: Returns a list of 5-model collections
             for each considered ML model type.
    """

    # split training dataframe randomly into five folds
    # random permutation of row indices
    rows = df_training.shape[0]
    rnd_indices = rng.permutation(rows)
    train_folds = [df_training.iloc[np.concatenate((rnd_indices[0:rows*i//num_folds],
                                                    rnd_indices[rows*(i+1)//num_folds:rows]))]
                   for i in range(num_folds)]
    test_folds = [df_training.iloc[rnd_indices[rows*i//num_folds:rows*(i+1)//num_folds]]
                  for i in range(num_folds)]

    # for each ML model type
    # train num_folds models with one fold as the test set,
    # and the remaining folds as the training set
    all_trained_models = []

    # xgboost, early_stopping_rounds=5, learning_rate=0.3
    xgb1_fold = []
    for i in range(num_folds):
        xgbmodel = xgb.XGBRegressor(early_stopping_rounds=5, learning_rate=0.3)

        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        xgbmodel.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=False)
        xgb1_fold.append(xgbmodel)
    all_trained_models.append(xgb1_fold)

    # xgboost, early_stopping_rounds=5, learning_rate=0.1
    xgb2_fold = []
    for i in range(num_folds):
        xgbmodel = xgb.XGBRegressor(early_stopping_rounds=5, learning_rate=0.1)

        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        xgbmodel.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=False)
        xgb2_fold.append(xgbmodel)
    all_trained_models.append(xgb2_fold)

    # xgboost, early_stopping_rounds=10, learning_rate=0.3
    xgb3_fold = []
    for i in range(num_folds):
        xgbmodel = xgb.XGBRegressor(early_stopping_rounds=10, learning_rate=0.3)

        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        xgbmodel.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=False)
        xgb3_fold.append(xgbmodel)
    all_trained_models.append(xgb3_fold)

    # xgboost, early_stopping_rounds=10, learning_rate=0.1
    xgb4_fold = []
    for i in range(num_folds):
        xgbmodel = xgb.XGBRegressor(early_stopping_rounds=10, learning_rate=0.1)

        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        xgbmodel.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=False)
        xgb4_fold.append(xgbmodel)
    all_trained_models.append(xgb4_fold)

    return all_trained_models


def predict_loads(models, df_selected):
    """ Models is a list containing
        the models obtained through 5-fold cross validation.
        Hence each individual "model" is actually a collection of 5 models,
        and its predictions are obtained as an average of predictions of all 5 models in the collection.

    :return: Load predictions based on the input columns in df_selected.
             Note that the returned structure is a python list of numpy arrays.
    """
    predictions = []
    for model_fold in models:
        sum_preds = np.zeros(df_selected.shape[0])
        for model in model_fold:
            preds = model.predict(df_selected[['dnorm', 'hnorm', 'diagnorm', 'area',
                                               'sine', 'cosine', 'd/h', 'h/d']])
            sum_preds += preds
        sum_preds /= len(model_fold)

        predictions.append(sum_preds)
    return predictions


def compute_cvrmse(loads, predictions):
    """ Compare the load array with each numpy array in the predictions list and
        compute CV(RMSE) = sqrt(sum_i (Y_i - Yhat_i)^2 / n)/ Ybar values,
        where Y are the actual (simulated) load values,
        Ybar is the average Y value, and
        Yhat are the predicted values.

        It is assumed that loads and predictions have the same number of elements.
        It is also assumed that Ybar is not zero!

    :param loads:       numpy array containing the actual (simulated) load values
    :param predictions: list of numpy arrays with the predicted load values
    :return:            list of cv(rmse) values for each prediction from the list
    """
    average_load = np.average(loads)
    rows = loads.shape[0]
    cvrmse = [math.sqrt(np.sum((loads - predictions[i])**2) / rows) / average_load
              for i in range(len(predictions))]
    return cvrmse


def process_this_combination(params):
    """ For the input columns and the output (load) column in df_selected,
        samples <sample_size> rows using the <sampling_method>,
        trains all ML model types on the selected sample using 5-fold cross validation,
        and computes the cv(rmse) for the prediction of all trained ML models on whole df_selected.
        This process is repeated <repeat_times> times,
        and the list with the average cvrmse values are returned.
    """
    df, climate, obstacle, orientation, heat_SP, cool_SP, load, sampling_method_name, num_inputs = params
    print(f'processing cl={climate}, ob={obstacle}, or={orientation}, '
          f'hsp={heat_SP}, csp={cool_SP}, load={load}, '
          f'sampling={sampling_method_name}, num_inp={num_inputs}')

    # select data for all overhang depths and heights
    df_selected = df[(df.climate == climate) &
                     (df.obstacle == obstacle) &
                     (df.orientation == orientation) &
                     (df.heat_SP == heat_SP) &
                     (df.cool_SP == cool_SP)]

    # keep only the columns necessary for training (plus depth and height as "indices")
    if num_inputs == 8:
        df_selected = df_selected[['height', 'depth', 'dnorm', 'hnorm', 'diagnorm', 'area',
                                   'sine', 'cosine', 'd/h', 'h/d', load]]
    else:  # num_inputs=2, so use six zero columns instead of additional inputs
        df_selected = df_selected[['height', 'depth', 'dnorm', 'hnorm', 'zeros1',
                                   'zeros2', 'zeros3', 'zeros4', 'zeros5', 'zeros6', load]]
        df_selected = df_selected.rename(columns={'zeros1': 'diagnorm', 'zeros2': 'area', 'zeros3': 'sine',
                                                  'zeros4': 'cosine', 'zeros5': 'd/h', 'zeros6': 'h/d'})

    # set height and depth as the multi-index
    # to easily extract all rows that correspond to sampled height/depth values
    df_selected = df_selected.set_index(['height', 'depth'])
    df_selected = df_selected.sort_index()

    # placeholder for the sums of cvrmse for each model type separately
    # this will later become the list of average values
    sum_cvrmse_results = np.zeros(num_models)

    # repeat necessary number of times
    for _ in range(repeat_times):

        # make a sample of height/depth values
        if sampling_method_name=='mipt':
            sample_init = mipt(sample_size)
        elif sampling_method_name=='mipt_full':
            sample_init = mipt_full(sample_size)
        else:
            sample_init = LHS_maximin(sample_size)

        # the length of height_range is 25, and the length of depth_range is 81
        height_range = np.linspace(0.01, 0.49, 25).round(decimals=2)
        depth_range = np.linspace(0.0, 1.6, 81).round(decimals=2)
        # while coordinates in the sample_init have values between 0.0 and 1.0 only
        sample = [(height_range[math.floor(25 * y)], depth_range[math.floor(81 * x)])
                  for [x, y] in sample_init]

        # extract the rows that correspond to sampled height/depth values
        # (note that sample MUST BE a list of tuples for .loc to work as expected)
        df_training = df_selected.loc[sample]

        # train all ML model types on the sampled rows by 5-fold cross validation
        models = train_models(df_training, load)

        # predict load values for all rows
        predictions = predict_loads(models, df_selected)

        # compute cvrmse values for all predictions at once
        cvrmse_results = compute_cvrmse(df_selected[load].to_numpy(), predictions)
        sum_cvrmse_results += cvrmse_results

    sum_cvrmse_results /= repeat_times
    return [climate, obstacle, orientation, heat_SP, cool_SP, load, sampling_method_name, num_inputs] \
        + sum_cvrmse_results.tolist()


def generate_params(df):
    # pass through all combinations of the office cell model parameters,
    # apart from the overhang depth and height
    for climate in [4]: # range(6):
        for obstacle in [0]: # range(5):
            for orientation in [0.0]: # [0.0, 45.0, -45.0]:
                for heat_SP in [21]: # [19, 21]:
                    for cool_SP in [24]: # [24, 26]:
                        for load in ['heat_load [kWh/m2]', 'cool_load [kWh/m2]', 'light_load [kWh/m2]', 'primary [kWh/m2]']:
                            for sampling_method_name in ['LHS_maximin', 'mipt', 'mipt_full']:
                                for num_inputs in [2, 8]:
                                    yield(df, climate, obstacle, orientation, heat_SP, cool_SP,
                                          load, sampling_method_name, num_inputs)


def process_all_combinations(df):
    """ Goes through all combinations of office cell model parameters,
        for each of them selects the corresponding part of the dataframe df (inputs+output),
        calls further method to obtain average(cvrmse) values
        computed through training all ML models for those inputs+output,
        and at the end saves all this information into an external csv file.
    """
    all_cvrmse_results = []

    # run the training for each parameter combination separately in parallel
    params = generate_params(df)
    with mp.Pool(max(mp.cpu_count(),1)) as pool:
        for cvrmse_results in pool.map(process_this_combination, params):
            all_cvrmse_results.append(cvrmse_results)

    # create a dataframe from all training results
    df_all_cvrmse_results = pd.DataFrame.from_records(data=all_cvrmse_results,
                                                      columns=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP',
                                                               'load', 'sampling_method', 'num_inputs',
                                                               #LATER: 'dnn881', 'dnn8551', 'dnn84441', 'dnn843331', 'dnn8333331', 'dnn83332221',
                                                               'xgb_early5_lr0.3', 'xgb_early5_lr0.1', 'xgb_early10_lr0.3', 'xgb_early10_lr0.1',
                                                               #LATER: 'gbdtpl'
                                                               ])
    timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    cvrmse_results_file = 'cvrmse_results_'+timestamp+'.csv'
    df_all_cvrmse_results.to_csv(cvrmse_results_file, index=False, float_format='%g')


if __name__=="__main__":
    df = pd.read_csv('collected_results.csv')
    mp.freeze_support()
    process_all_combinations(df)
