import numpy as np
import pandas as pd
# import tensorflow as tf
import torch
import torch.nn as nn
import datetime
import math
import multiprocessing as mp
import os

from mipt import mipt, mipt_full, LHS_maximin

sample_size = 100   # the number of samples to be selected
repeat_times = 5    # the number of ML models to be trained for each parameter combination,
                    # keeping only their average cvrmse as the final result
num_models = 4      # the number of distinct ML models used in training
                    # at the moment, they are:
                    # dnn variants: dnn881, dnn8551, dnn8631, dnn843331
                    # xbg variants: xgb-early10-lr0.3, xgb-early10-lr0.1, xgb-early10-lr0.03
num_folds = 5       # number of folds for cross validation

rng = np.random.default_rng()   # random number generator,
                                # used for random splitting in 5-fold cross validation


# re-implementation of tensorflow's early stopping
class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model_dict = None

    def should_stop(self, validation_loss, model):
        # in case no progress is ever made, keep track of the weights from the first epoch
        if self.best_model_dict is None:
            self.best_model_dict = model.state_dict()

        self.counter += 1
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.best_model_dict = model.state_dict()
            self.counter = 0
        return self.counter >= self.patience

    def restore_weights(self, model):
        model.load_state_dict(self.best_model_dict)

    def get_best_model_dict(self):
        return self.best_model_dict


def train_dnn_fold(load, train_folds, test_folds, neurons):
    """ Auxiliary method for training dnn based models with torch.
    """
    dnn_fold = []
    for i in range(num_folds):
        # print(f'training dnn{neurons} - fold {i}...')

        # the first layer with its input shape
        dnn_model = nn.Sequential(nn.Linear(8, neurons[0]),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.BatchNorm1d(neurons[0]))
        # the interior layers
        for i in range(1, len(neurons)):
            dnn_model.append(nn.Linear(neurons[i-1], neurons[i]))
            dnn_model.append(nn.ReLU())
            dnn_model.append(nn.Dropout(0.3))
            dnn_model.append(nn.BatchNorm1d(neurons[i]))
        # the last layer
        dnn_model.append(nn.Linear(neurons[-1], 1))

        # other model training parameters
        objective = nn.MSELoss()
        optimizer = torch.optim.Adam(params=dnn_model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        # prepare training and test sets
        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']].to_numpy(dtype=np.float32)
        X_train_t = torch.from_numpy(X_train)
        y_train = train_folds[i][load].to_numpy(dtype=np.float32)
        y_train_t = torch.from_numpy(y_train)
        y_train_t = torch.unsqueeze(y_train_t, -1)      # to appropriately transform the shape
        train_set = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']].to_numpy(dtype=np.float32)
        X_test_t = torch.from_numpy(X_test)
        y_test = test_folds[i][load].to_numpy(dtype=np.float32)
        y_test_t = torch.from_numpy(y_test)
        y_test_t = torch.unsqueeze(y_test_t, -1)
        test_set = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)

        # training/validation loop
        for epoch in range(1000):
            # train the model
            dnn_model.train()
            total_train_loss = 0.0
            for idx, (X_train_t, y_train_t) in enumerate(train_loader):
                optimizer.zero_grad()
                pred_t = dnn_model(X_train_t)
                loss = objective(pred_t, y_train_t)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            total_train_loss /= idx+1

            # validate the model
            dnn_model.eval()
            total_val_loss = 0.0
            for idx, (X_test_t, y_test_t) in enumerate(test_loader):
                pred_t = dnn_model(X_test_t)
                val_loss = objective(pred_t, y_test_t)
                total_val_loss += val_loss.item()
            total_val_loss /= idx+1

            # early stopping?
            if early_stopping.should_stop(total_val_loss, dnn_model):
                early_stopping.restore_weights(dnn_model)
                # dnn_model.load_state_dict(early_stopping.get_best_model_dict())
                break

            # update learning rate, if necessary
            scheduler.step()

        # add the model to the fold
        dnn_fold.append(dnn_model)

    return dnn_fold


def train_dnn_fold_tensorflow(load, train_folds, test_folds, neurons):
    """ Auxiliary method for training dnn based models in tensorflow.
    """
    dnn_fold = []
    for i in range(num_folds):
        # the first layer with its input shape
        layers = [tf.keras.layers.Dense(units=neurons[0], activation='relu', input_shape=[8]),
                  tf.keras.layers.Dropout(rate=0.3),
                  tf.keras.layers.BatchNormalization()]
        # the middle layers
        for layer_width in neurons[1:]:
            layers = layers + [tf.keras.layers.Dense(units=layer_width, activation='relu'),
                               tf.keras.layers.Dropout(rate=0.3),
                               tf.keras.layers.BatchNormalization()]
        # the last layer
        layers = layers + [tf.keras.layers.Dense(units=1)]

        # dnn model structure, optimizer and loss function
        dnn_model = tf.keras.Sequential(layers)
        dnn_model.compile(optimizer='adam',
                          loss='mse')

        # training and test sets
        X_train = train_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_train = train_folds[i][load]
        X_test = test_folds[i][['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']]
        y_test = test_folds[i][load]

        # set up early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            min_delta=0.001,
            patience=10,        # number of epochs to wait before stopping training
            restore_best_weights=True,
            # start_from_epoch=10
        )

        # fit the model
        dnn_model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=10,
                      epochs=1000,
                      callbacks=[early_stopping],
                      verbose=0)

        # add the model to the fold
        dnn_fold.append(dnn_model)

    return dnn_fold


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

    # dnn-based models with dropout and batch normalization
    # all_trained_models.append(train_dnn_fold_tensorflow(load, train_folds, test_folds, neurons=[8]))
    # all_trained_models.append(train_dnn_fold_tensorflow(load, train_folds, test_folds, neurons=[6, 3]))
    # all_trained_models.append(train_dnn_fold_tensorflow(load, train_folds, test_folds, neurons=[4, 3, 3, 3]))

    all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[8]))
    all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[5, 5]))
    all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[6, 3]))
    # all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[4, 4, 4]))
    all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[4, 3, 3, 3]))
    # all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[3, 3, 3, 3, 3]))
    # all_trained_models.append(train_dnn_fold(load, train_folds, test_folds, neurons=[3, 3, 3, 2, 2, 2]))

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
            # prediction with tensorflow
            # preds = model.predict(df_selected[['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']])

            # prediction with pytorch is a bit low level...
            model.eval()
            input = df_selected[['dnorm', 'hnorm', 'diagnorm', 'area', 'sine', 'cosine', 'd/h', 'h/d']].to_numpy(dtype=np.float32)
            input_t = torch.from_numpy(input)
            preds = model(input_t).detach().numpy()
            # gets rid of the additional dimension
            preds = preds.reshape((df_selected.shape[0],))

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
    for rtc in range(repeat_times):
        print(f'  repeat times counter={rtc}...')

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
                for heat_SP in [19]: # [19, 21]:
                    for cool_SP in [24]: # [24, 26]:
                        for load in ['heat_load [kWh/m2]', 'cool_load [kWh/m2]', 'light_load [kWh/m2]', 'primary [kWh/m2]']:
                            for sampling_method_name in ['mipt_full']: # ['LHS_maximin', 'mipt', 'mipt_full']:
                                for num_inputs in [2]: # [2, 8]:
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
                                                               'dnn881', 'dnn8551', 'dnn8631', 'dnn843331',
                                                               ])
    timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    cvrmse_results_file = 'cvrmse_results_'+timestamp+'.csv'
    df_all_cvrmse_results.to_csv(cvrmse_results_file, index=False, float_format='%g')


if __name__=="__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(f'loading simulation data...')
    df = pd.read_csv('collected_results.csv')
    mp.freeze_support()
    process_all_combinations(df)
