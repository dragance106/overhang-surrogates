"""
Unlike train_xgb.py, which trains xgb models for Subsections 3.2.1-3.2.4,
this file serves to train a selection of xgb_lr0.1 models for Subsection 3.2.5
with 8 inputs and sample sizes 12, 25, 50, 100.
The general structure of both files is relatively similar,
with small differences in model variants and iterated parameters.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import math
import multiprocessing as mp

from mipt import mipt_full

repeat_times = 5    # the number of ML models to be trained for each parameter combination,
                    # keeping only their average cvrmse as the final result
num_models = 4      # the number of distinct ML models used in training:
                    # xgb100, xgb50, xgb25 and xgb12 with 8 inputs
num_folds = 5       # number of folds for cross validation

rng = np.random.default_rng()   # random number generator,
                                # used for random splitting in 5-fold cross validation

def train_xgb_fold(load, df_selected, sample, sample_size=100):
    """ Auxiliary method for training xgb models with 5-fold cross validation.
    Each xgb "model" is actually a collection/list of 5 models
    that are trained on different folds of df_training as their test sets
    (with the remaining folds used as training sets).

    :return: Returns a list of 5-model collections
             for each considered ML model type.
    """

    # extract the rows that correspond to sampled height/depth values
    # (note that sample MUST BE a list of (tuples) for .loc to work as expected)
    # the FIRST sample_size elements of the sample are taken into account
    df_training = df_selected.loc[sample[:sample_size]]

    # random permutation of row indices
    rows = df_training.shape[0]
    rnd_indices = rng.permutation(rows)
    # split training dataframe randomly into five folds
    train_folds = [df_training.iloc[np.concatenate((rnd_indices[0:rows*i//num_folds],
                                                    rnd_indices[rows*(i+1)//num_folds:rows]))]
                   for i in range(num_folds)]
    test_folds = [df_training.iloc[rnd_indices[rows*i//num_folds:rows*(i+1)//num_folds]]
                  for i in range(num_folds)]

    xgb_fold = []
    for i in range(num_folds):
        xgb_model = xgb.XGBRegressor(early_stopping_rounds=10, learning_rate=0.1)

        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        xgb_fold.append(xgb_model)
    return xgb_fold


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
    if abs(average_load) < 1e-6:
        average_load = 1.0
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
    df, climate, obstacle, orientation, heat_SP, cool_SP, load = params
    print(f'processing cl={climate}, ob={obstacle}, or={orientation}, '
          f'hsp={heat_SP}, csp={cool_SP}, load={load}')

    # select data for all overhang depths and heights
    df_selected = df[(df.climate == climate) &
                     (df.obstacle == obstacle) &
                     (df.orientation == orientation) &
                     (df.heat_SP == heat_SP) &
                     (df.cool_SP == cool_SP)]

    # keep only the columns necessary for training (plus depth and height as "indices")
    df_selected = df_selected[['height', 'depth', 'dnorm', 'hnorm', 'diagnorm', 'area',
                               'sine', 'cosine', 'd/h', 'h/d', load]]

    # set height and depth as the multi-index
    # to easily extract all rows that correspond to sampled height/depth values
    df_selected = df_selected.set_index(['height', 'depth'])
    df_selected = df_selected.sort_index()

    # placeholder for the sums of cvrmse for each model type separately
    # this will later become the list of average values
    sum_cvrmse_results = np.zeros(num_models)

    # repeat necessary number of times
    for rtc in range(repeat_times):
        print(f'  repeat times counter={rtc}...')

        # make a sample of height/depth values
        sample_init = mipt_full(100)
        # due to the iterative nature of mipt_full
        # the FIRST k points of sample_init also serve as a smaller sample of size k

        # the length of height_range is 25, and the length of depth_range is 81,
        # while coordinates in the sample_init have values between 0.0 and 1.0 only
        height_range = np.linspace(0.01, 0.49, 25).round(decimals=2)
        depth_range = np.linspace(0.0, 1.6, 81).round(decimals=2)
        sample = [(height_range[math.floor(25 * y)], depth_range[math.floor(81 * x)])
                  for [x, y] in sample_init]

        # train all xgboost models on the sampled rows by 5-fold cross validation
        models = [train_xgb_fold(load, df_selected, sample, sample_size=12),
                  train_xgb_fold(load, df_selected, sample, sample_size=25),
                  train_xgb_fold(load, df_selected, sample, sample_size=50),
                  train_xgb_fold(load, df_selected, sample, sample_size=100)]

        # predict load values for all rows
        predictions = predict_loads(models, df_selected)

        # compute cvrmse values for all predictions at once
        cvrmse_results = compute_cvrmse(df_selected[load].to_numpy(), predictions)
        sum_cvrmse_results += cvrmse_results

    sum_cvrmse_results /= repeat_times
    return [climate, obstacle, orientation, heat_SP, cool_SP, load] + sum_cvrmse_results.tolist()


def generate_params(df):
    # pass through all combinations of the office cell model parameters,
    # apart from the overhang depth and height
    for climate in range(6):
        for obstacle in range(5):
            for orientation in [0.0, 45.0, -45.0]:
                for heat_SP in [19, 21]:
                    for cool_SP in [24, 26]:
                        for load in ['heat_load [kWh/m2]', 'cool_load [kWh/m2]', 'light_load [kWh/m2]', 'primary [kWh/m2]']:
                            yield(df, climate, obstacle, orientation, heat_SP, cool_SP, load)


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
    with mp.Pool(max(mp.cpu_count(), 1)) as pool:
        for cvrmse_results in pool.map(process_this_combination, params):
            all_cvrmse_results.append(cvrmse_results)

    # create a dataframe from all training results
    df_all_cvrmse_results = pd.DataFrame.from_records(data=all_cvrmse_results,
                                                      columns=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load',
                                                               'xgb12_8', 'xgb25_8', 'xgb50_8', 'xgb100_8'])
    timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    cvrmse_results_file = 'cvrmse_xgb_8_inputs_'+timestamp+'.csv'
    df_all_cvrmse_results.to_csv(cvrmse_results_file, index=False, float_format='%g')


def train_xgb_models_for_starting_case():
    """
    This method trains four xgb models for the starting case of
    New York climate, no obstacles, south orientation, hsp=21 and csp=24,
    and saves their predictions next to simulated values
    for later visualization with vedo and/or seaborn heat maps.
    """

    # Select the data for the starting case
    print(f'loading simulation data...')
    df = pd.read_csv('collected_results.csv')
    df = df[(df.climate==4) &
            (df.obstacle==0) &
            (df.orientation==0) &
            (df.heat_SP==21) &
            (df.cool_SP==24)].copy()

    # Set height and depth as multiindex
    # to easily extract all rows that correspond to sampled height/depth values
    df = df.set_index(['height', 'depth'])
    df = df.sort_index()

    # make a sample of height/depth values
    sample_init = mipt_full(100)
    height_range = np.linspace(0.01, 0.49, 25).round(decimals=2)
    depth_range = np.linspace(0.0, 1.6, 81).round(decimals=2)
    sample = [(height_range[math.floor(25 * y)], depth_range[math.floor(81 * x)])
              for [x, y] in sample_init]

    # train xgboost models for each load by 5-fold cross validation on the sampled rows
    # each train_xgb_fold in the following lists
    # actually gives a further list of five xgb models obtained by 5-fold cross validation
    print(f'training heating load models...')
    models_heat = [train_xgb_fold('heat_load [kWh/m2]', df, sample, sample_size=12),
                   train_xgb_fold('heat_load [kWh/m2]', df, sample, sample_size=25),
                   train_xgb_fold('heat_load [kWh/m2]', df, sample, sample_size=50),
                   train_xgb_fold('heat_load [kWh/m2]', df, sample, sample_size=100)]

    print(f'training cooling load models...')
    models_cool = [train_xgb_fold('cool_load [kWh/m2]', df, sample, sample_size=12),
                   train_xgb_fold('cool_load [kWh/m2]', df, sample, sample_size=25),
                   train_xgb_fold('cool_load [kWh/m2]', df, sample, sample_size=50),
                   train_xgb_fold('cool_load [kWh/m2]', df, sample, sample_size=100)]

    print(f'training lighting load models...')
    models_light = [train_xgb_fold('light_load [kWh/m2]', df, sample, sample_size=12),
                    train_xgb_fold('light_load [kWh/m2]', df, sample, sample_size=25),
                    train_xgb_fold('light_load [kWh/m2]', df, sample, sample_size=50),
                    train_xgb_fold('light_load [kWh/m2]', df, sample, sample_size=100)]

    print(f'training primary energy models...')
    models_primary = [train_xgb_fold('primary [kWh/m2]', df, sample, sample_size=12),
                      train_xgb_fold('primary [kWh/m2]', df, sample, sample_size=25),
                      train_xgb_fold('primary [kWh/m2]', df, sample, sample_size=50),
                      train_xgb_fold('primary [kWh/m2]', df, sample, sample_size=100)]

    # predict load values for the whole starting case
    # note that each preds_... is a python list of numpy arras
    print(f'predicting load values for all height/depth pairs...')
    preds_heat = predict_loads(models_heat, df)
    preds_cool = predict_loads(models_cool, df)
    preds_light = predict_loads(models_light, df)
    preds_primary = predict_loads(models_primary, df)

    # now add these predictions as new columns to df
    df['pred_heat_xgb12'] = preds_heat[0]
    df['pred_heat_xgb25'] = preds_heat[1]
    df['pred_heat_xgb50'] = preds_heat[2]
    df['pred_heat_xgb100'] = preds_heat[3]

    df['pred_cool_xgb12'] = preds_cool[0]
    df['pred_cool_xgb25'] = preds_cool[1]
    df['pred_cool_xgb50'] = preds_cool[2]
    df['pred_cool_xgb100'] = preds_cool[3]

    df['pred_light_xgb12'] = preds_light[0]
    df['pred_light_xgb25'] = preds_light[1]
    df['pred_light_xgb50'] = preds_light[2]
    df['pred_light_xgb100'] = preds_light[3]

    df['pred_primary_xgb12'] = preds_primary[0]
    df['pred_primary_xgb25'] = preds_primary[1]
    df['pred_primary_xgb50'] = preds_primary[2]
    df['pred_primary_xgb100'] = preds_primary[3]

    # drop the columns we no longer need
    df = df.drop(columns=['heat_load [J]', 'cool_load [J]', 'light_load [J]',
                          'dnorm', 'hnorm', 'diagonal', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d',
                          'zeros1', 'zeros2', 'zeros3', 'zeros4', 'zeros5', 'zeros6'])

    # finally save everything to a new csv file
    print(f'saving predicted loads...')
    df.to_csv('starting_case_predictions.csv', index=True, float_format='%g')

    # also save the information on sampled points
    df_sample = pd.DataFrame.from_records(data=sample, columns=['height', 'depth'])
    df_sample.to_csv('starting_case_sample_set.csv', index=False, float_format='%g')

    print(f'done.')


if __name__=="__main__":
    # print(f'loading simulation results...')
    # df = pd.read_csv('collected_results.csv')
    # mp.freeze_support()
    # process_all_combinations(df)

    train_xgb_models_for_starting_case()