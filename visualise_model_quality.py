import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from vedo import *
import colorcet


def join_model_results():
    """
    Auxiliary method to join pandas dataframes containing cvrmse results for different ML models.
    """
    df1 = pd.read_csv('cvrmse_dnn881_results.csv')
    df1 = df1.set_index(['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df2 = pd.read_csv('cvrmse_dnn8551_results.csv')
    df2 = df2.set_index(['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df3 = df1.merge(df2, on=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df4 = pd.read_csv('cvrmse_dnn84441_results.csv')
    df4 = df4.set_index(['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df5 = df3.merge(df4, on=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df6 = pd.read_csv('cvrmse_xgboost_results.csv')
    df6 = df6.set_index(['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])

    df7 = df5.merge(df6, on=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP', 'load', 'sampling_method', 'num_inputs'])
    df7.sort_index()
    df7.to_csv('cvrmse_results.csv', index=False, float_format='%g')


def reorganize_cvrmse_results():
    print(f'loading cvrmse results...')
    df = pd.read_csv('cvrmse_results.csv')

    small_df_list = []
    model_list = ['dnn881', 'dnn8551', 'dnn84441', 'xgb_lr0.3', 'xgb_lr0.1', 'xgb_lr0.03']

    for model in model_list:
        print(f'processing {model}...')

        small_df = df.copy()
        small_df = small_df.rename(columns={model: 'cvrmse'})
        small_df['model'] = model
        other_models = [item for item in model_list if item!=model]
        small_df = small_df.drop(columns=other_models)
        small_df_list.append(small_df)

    # now put all separate small dataframes together
    df_final = pd.concat(small_df_list)
    df_final.to_csv('cvrmse_predicted_results.csv', index=False, float_format='%g')


def reorganize_cvrmse_results_2():
    print(f'loading cvrmse results...')
    df = pd.read_csv('cvrmse_xgb_special.csv')

    small_df_list = []
    model_list = ['xgb12','xgb25','xgb50','xgb100']

    for model in model_list:
        print(f'processing {model}...')

        small_df = df.copy()
        small_df = small_df.rename(columns={model: 'cvrmse'})
        small_df['model'] = model
        other_models = [item for item in model_list if item!=model]
        small_df = small_df.drop(columns=other_models)
        small_df_list.append(small_df)

    # now put all separate small dataframes together
    df_final = pd.concat(small_df_list)
    df_final.to_csv('cvrmse_xgb_predicted_special.csv', index=False, float_format='%g')


def compute_cv_qcd_simulated_results():
    df = pd.read_csv('collected_results.csv')

    # group by the office cell model parameters
    df = df.set_index(['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP'])
    df = df.sort_index()

    new_rows = []
    # iterate through the index
    for i in df.index.unique():
        print(f'{i}...')
        # select the subset you need
        df_case = df.loc[i]

        # now compute std, mean, CV, Q1, Q3 and QCD for each load
        for load in ['heat_load [kWh/m2]', 'cool_load [kWh/m2]', 'light_load [kWh/m2]', 'primary [kWh/m2]']:
            m = df_case[load].mean()
            s = df_case[load].std()
            if abs(m) > 1e-6:
                cv = s/m
            else:
                cv = s

            q1 = df_case[load].quantile(.25)
            q3 = df_case[load].quantile(.75)
            if abs(q3+q1) > 1e-6:
                qcd = (q3-q1)/(q3+q1)
            else:
                qcd = 0.0

            new_rows.append(list(i) + [load, cv, qcd])

    new_df = pd.DataFrame.from_records(data = new_rows,
                                       columns=['climate', 'obstacle', 'orientation', 'heat_SP', 'cool_SP',
                                                'load', 'cv', 'qcd'])
    new_df.to_csv('cv_qcd_simulated_results.csv', index=False, float_format='%g')


def draw_barplot_D1():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df     # dfnp = df[(df.load!='primary [kWh/m2]')]
    # SEPARATE PLOTS FOR DNN AND XGBOOST MODELS
    # dfnp = dfnp[(dfnp.model == 'xgb_lr0.3') |
    #             (dfnp.model == 'xgb_lr0.1') |
    #             (dfnp.model == 'xgb_lr0.03')]
    dfnp = dfnp[(dfnp.model == 'dnn881') |
                (dfnp.model == 'dnn8551') |
                (dfnp.model == 'dnn84441')]
    dfnp = dfnp.replace({'cool_load [kWh/m2]': 'Cooling load',
                         'heat_load [kWh/m2]': 'Heating load',
                         'light_load [kWh/m2]': 'Lighting load',
                         'primary [kWh/m2]': 'Primary energy'})
    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'bar',
                    x = 'model',
                    y = 'cvrmse',
                    hue = 'num_inputs',
                    col = 'load',
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    alpha = 0.75,
                    height = 5,
                    aspect = 1.25)
    # show bar labels
    for ax in g.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, labels=[f'{v.get_height():.3f}' for v in c])
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("ML model", "Average CV(RMSE)")
    g.legend.set_title("No. inputs")
    g.savefig('fig_models/d1_barplot_dnn.png')


def draw_boxplot_D1():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df   # dfnp = df[(df.load!='primary [kWh/m2]')]
    # SEPARATE PLOTS FOR DNN AND XGBOOST MODELS
    dfnp = dfnp[(dfnp.model == 'xgb_lr0.3') |
                (dfnp.model == 'xgb_lr0.1') |
                (dfnp.model == 'xgb_lr0.03')]
    # dfnp = dfnp[(dfnp.model == 'dnn881') |
    #             (dfnp.model == 'dnn8551') |
    #             (dfnp.model == 'dnn84441')]
    dfnp = dfnp.replace({'cool_load [kWh/m2]': 'Cooling load',
                         'heat_load [kWh/m2]': 'Heating load',
                         'light_load [kWh/m2]': 'Lighting load',
                         'primary [kWh/m2]': 'Primary energy'})
    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'box',
                    x = 'model',
                    y = 'cvrmse',
                    hue = 'num_inputs',
                    col = 'load',
                    showfliers = False,
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.75)
    # show bar labels
    # for ax in g.axes.ravel():
    #     for c in ax.containers:
    #         ax.bar_label(c, labels=[f'{v.get_height():.3f}' for v in c])
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    g.legend.set_title("No. inputs")
    g.savefig('fig_models/d1_boxplot_xgboost.png')


def draw_barplot_D3():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df   # dfnp = df[(df.load!='primary [kWh/m2]')]
    # SEPARATE PLOTS FOR DNN AND XGBOOST MODELS
    # dfnp = dfnp[(dfnp.model == 'xgb_lr0.3') |
    #             (dfnp.model == 'xgb_lr0.1') |
    #             (dfnp.model == 'xgb_lr0.03')]
    dfnp = dfnp[(dfnp.model == 'dnn881') |
                (dfnp.model == 'dnn8551') |
                (dfnp.model == 'dnn84441')]
    dfnp = dfnp[((dfnp.model == 'xgb_lr0.3') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.1') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.03') & (dfnp.num_inputs == 8))|
                ((dfnp.model == 'dnn881') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn8551') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn84441') & (dfnp.num_inputs == 2))]
    dfnp = dfnp.replace({'cool_load [kWh/m2]': 'Cooling load',
                         'heat_load [kWh/m2]': 'Heating load',
                         'light_load [kWh/m2]': 'Lighting load'})
    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'bar',
                    x = 'model',
                    y = 'cvrmse',
                    hue = 'sampling_method',
                    col = 'load',
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    alpha = 0.75,
                    height = 5,
                    aspect = 1.25)
    # show bar labels
    for ax in g.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, labels=[f'{v.get_height():.3f}' for v in c])
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("ML model", "Average CV(RMSE)")
    g.legend.set_title("Sampling method")
    g.savefig('fig_models/d3_barplot_dnn.png')


def draw_boxplot_D3():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df    # dfnp = df[(df.load!='primary [kWh/m2]')]
    # SEPARATE PLOTS FOR DNN AND XGBOOST MODELS
    # dfnp = dfnp[(dfnp.model == 'xgb_lr0.3') |
    #             (dfnp.model == 'xgb_lr0.1') |
    #             (dfnp.model == 'xgb_lr0.03')]
    dfnp = dfnp[(dfnp.model == 'dnn881') |
                (dfnp.model == 'dnn8551') |
                (dfnp.model == 'dnn84441')]
    dfnp = dfnp[((dfnp.model == 'xgb_lr0.3') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.1') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.03') & (dfnp.num_inputs == 8))|
                ((dfnp.model == 'dnn881') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn8551') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn84441') & (dfnp.num_inputs == 2))]
    dfnp = dfnp.replace({'cool_load [kWh/m2]': 'Cooling load',
                         'heat_load [kWh/m2]': 'Heating load',
                         'light_load [kWh/m2]': 'Lighting load',
                         'primary [kWh/m2]': 'Primary energy'})
    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'box',
                    x = 'model',
                    y = 'cvrmse',
                    hue = 'sampling_method',
                    col = 'load',
                    showfliers = False,
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.75)
    # show bar labels
    # for ax in g.axes.ravel():
    #     for c in ax.containers:
    #         ax.bar_label(c, labels=[f'{v.get_height():.3f}' for v in c])
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    g.legend.set_title("Sampling method")
    g.savefig('fig_models/d3_boxplot_dnn.png')


def draw_boxplot_D4():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df    # dfnp = df[(df.load!='primary [kWh/m2]')]
    dfnp = dfnp[((dfnp.model == 'xgb_lr0.3') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.1') & (dfnp.num_inputs == 8)) |
                ((dfnp.model == 'xgb_lr0.03') & (dfnp.num_inputs == 8))|
                ((dfnp.model == 'dnn881') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn8551') & (dfnp.num_inputs == 2)) |
                ((dfnp.model == 'dnn84441') & (dfnp.num_inputs == 2))]
    dfnp = dfnp[dfnp.sampling_method == 'mipt_full']
    dfnp = dfnp.replace({'cool_load [kWh/m2]': 'Cooling load',
                         'heat_load [kWh/m2]': 'Heating load',
                         'light_load [kWh/m2]': 'Lighting load',
                         'primary [kWh/m2]': 'Primary energy'})
    dfnp_heat = dfnp[dfnp.load == 'Heating load']
    dfnp_cool = dfnp[dfnp.load == 'Cooling load']
    dfnp_light = dfnp[dfnp.load == 'Lighting load']
    dfnp_primary = dfnp[dfnp.load == 'Primary energy']

    sns.set_theme(style="whitegrid", font_scale=1.6)

    gh = sns.catplot(data = dfnp_heat,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # hue = 'sampling_method',
                     # col = 'load',
                     showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.9)
    # set y-axis limits to restrict showing of far outliers for DNN models
    # for ax in gh.axes.ravel():
    #     ax.set_ylim(0, 0.36)
    gh.despine(left = True)
    gh.set_titles('{col_name}')
    gh.fig.suptitle('Heating load')
    gh.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    gh.savefig('fig_models/d4_boxplot_heat.png')

    gc = sns.catplot(data = dfnp_cool,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # hue = 'sampling_method',
                     # col = 'load',
                     showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.9)
    # set y-axis limits to restrict showing of far outliers for DNN models
    # for ax in gc.axes.ravel():
    #     ax.set_ylim(0, 0.4)
    gc.despine(left = True)
    gc.set_titles('{col_name}')
    gc.fig.suptitle('Cooling load')
    gc.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    gc.savefig('fig_models/d4_boxplot_cool.png')

    gl = sns.catplot(data = dfnp_light,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # hue = 'sampling_method',
                     # col = 'load',
                     showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.9)
    # set y-axis limits to restrict showing of far outliers for DNN models
    # for ax in gl.axes.ravel():
    #     ax.set_ylim(0, 0.15)
    gl.despine(left = True)
    gl.set_titles('{col_name}')
    gl.fig.suptitle('Lighting load')
    gl.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    # g.legend.set_title("Sampling method")
    gl.savefig('fig_models/d4_boxplot_light.png')

    gp = sns.catplot(data = dfnp_primary,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # hue = 'sampling_method',
                     # col = 'load',
                     showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.9)
    # set y-axis limits to restrict showing of far outliers for DNN models
    # for ax in gh.axes.ravel():
    #     ax.set_ylim(0, 0.36)
    gp.despine(left = True)
    gp.set_titles('{col_name}')
    gp.fig.suptitle('Primary energy')
    gp.set_axis_labels("ML model", "Distribution of CV(RMSE)")
    gp.savefig('fig_models/d4_boxplot_primary.png')


def draw_boxplot_D5_climate_renew():
    # prepare predicted results
    dfp = pd.read_csv('cvrmse_predicted_results.csv')
    # dfp = dfp[dfp.load!='primary [kWh/m2]']    # ONLY FOR XGB_LR0.1 MODEL WITH 8 INPUTS AND MIPT_FULL SAMPLING
    dfp = dfp[(dfp.model == 'xgb_lr0.1') &
              (dfp.num_inputs == 8) &
              (dfp.sampling_method == 'mipt_full')]
    # replace within the load column: cryptic
    dfp['load'] = dfp['load'].replace({'cool_load [kWh/m2]': 'Cooling load',
                                       'heat_load [kWh/m2]': 'Heating load',
                                       'light_load [kWh/m2]': 'Lighting load',
                                       'primary [kWh/m2]': 'Primary energy'})
    # replace within the climate column: numbers with city names
    dfp['climate'] = dfp['climate'].replace({0: 'Dubai',
                                             1: 'Honolulu',
                                             2: 'Tucson',
                                             3: 'San Diego',
                                             4: 'New York',
                                             5: 'Denver'})
    # create three plots in one facetgrid
    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfp,
                    kind = 'box',
                    x = 'climate',
                    y = 'cvrmse',
                    col = 'load',
                    # showfliers=False,
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.8)
    g.despine(left = True)
    g.set_titles('{col_name}')
    # g.set_axis_labels("Climate", "Distribution of CV(RMSE)")
    g.set_axis_labels("", "Distribution of CV(RMSE)")

    # now prepare simulated results
    dfs = pd.read_csv('cv_qcd_simulated_results.csv')
    # dfs = dfs[dfs.load!='primary [kWh/m2]']

    # draw each small box plot as the inset in one of the large box plots
    axins = [inset_axes(ax, '30%', '30%', loc='upper right', borderpad=0) for ax in g.axes.ravel()]
    for ax in axins:
        ax.tick_params(labelbottom=False)
    sns.boxplot(data=dfs[dfs.load=='heat_load [kWh/m2]'],
                x='climate',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[0])
    sns.boxplot(data=dfs[dfs.load=='cool_load [kWh/m2]'],
                x='climate',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[1])
    sns.boxplot(data=dfs[dfs.load=='light_load [kWh/m2]'],
                x='climate',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[2])
    sns.boxplot(data=dfs[dfs.load == 'primary [kWh/m2]'],
                x='climate',
                y='cv',
                showfliers=False,
                saturation=0.75,
                ax=axins[3])

    g.savefig('fig_models/d5_boxplot_climate_renew.png')


def draw_boxplot_D5_obstacles_renew():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df    # dfnp = df[(df.load!='primary [kWh/m2]')]
    dfnp = dfnp[(dfnp.model == 'xgb_lr0.1') &
                (dfnp.num_inputs == 8) &
                (dfnp.sampling_method == 'mipt_full')]
    # replace within the load column: cryptic
    dfnp['load'] = dfnp['load'].replace({'cool_load [kWh/m2]': 'Cooling load',
                                         'heat_load [kWh/m2]': 'Heating load',
                                         'light_load [kWh/m2]': 'Lighting load',
                                         'primary [kWh/m2]': 'Primary energy'})
    dfnp['obstacle'] = dfnp['obstacle'].replace({0: 'none',
                                                 1: 'med.\n11-14h',
                                                 2: 'high\n11-14h',
                                                 3: 'med.\n15-17h',
                                                 4: 'med.\n8-10h & 15-17h'})

    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'box',
                    x = 'obstacle',
                    y = 'cvrmse',
                    col = 'load',
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.8)
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("", "Distribution of CV(RMSE)")

    # now prepare simulated results
    dfs = pd.read_csv('cv_qcd_simulated_results.csv')
    # dfs = dfs[dfs.load!='primary [kWh/m2]']

    # draw each small box plot as the inset in one of the large box plots
    axins = [inset_axes(ax, '30%', '30%', loc='upper right', borderpad=0) for ax in g.axes.ravel()]
    for ax in axins:
        ax.tick_params(labelbottom=False)
    sns.boxplot(data=dfs[dfs.load=='heat_load [kWh/m2]'],
                x='obstacle',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[0])
    sns.boxplot(data=dfs[dfs.load=='cool_load [kWh/m2]'],
                x='obstacle',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[1])
    sns.boxplot(data=dfs[dfs.load=='light_load [kWh/m2]'],
                x='obstacle',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[2])
    sns.boxplot(data=dfs[dfs.load=='primary [kWh/m2]'],
                x='obstacle',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[3])

    g.savefig('fig_models/d5_boxplot_obstacles_renew.png')


def draw_boxplot_D5_orientation_renew():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df    # dfnp = df[(df.load!='primary [kWh/m2]')]
    dfnp = dfnp[(dfnp.model == 'xgb_lr0.1') &
                (dfnp.num_inputs == 8) &
                (dfnp.sampling_method == 'mipt_full')]
    dfnp['load'] = dfnp['load'].replace({'cool_load [kWh/m2]': 'Cooling load',
                                         'heat_load [kWh/m2]': 'Heating load',
                                         'light_load [kWh/m2]': 'Lighting load',
                                         'primary [kWh/m2]': 'Primary energy'})
    dfnp['orientation'] = dfnp['orientation'].replace({0: 'south',
                                                       45: 'south-east',
                                                       -45: 'south-west'})

    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'box',
                    x = 'orientation',
                    y = 'cvrmse',
                    col = 'load',
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.8)
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("", "Distribution of CV(RMSE)")

    # now prepare simulated results
    dfs = pd.read_csv('cv_qcd_simulated_results.csv')
    # dfs = dfs[dfs.load!='primary [kWh/m2]']

    # draw each small box plot as the inset in one of the large box plots
    axins = [inset_axes(ax, '30%', '30%', loc='upper right', borderpad=0) for ax in g.axes.ravel()]
    for ax in axins:
        ax.tick_params(labelbottom=False)
    sns.boxplot(data=dfs[dfs.load=='heat_load [kWh/m2]'],
                x='orientation',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[0])
    sns.boxplot(data=dfs[dfs.load=='cool_load [kWh/m2]'],
                x='orientation',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[1])
    sns.boxplot(data=dfs[dfs.load=='light_load [kWh/m2]'],
                x='orientation',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[2])
    sns.boxplot(data=dfs[dfs.load=='primary [kWh/m2]'],
                x='orientation',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[3])

    g.savefig('fig_models/d5_boxplot_orientation_renew.png')


def draw_boxplot_D5_setpoints_renew():
    df = pd.read_csv('cvrmse_predicted_results.csv')
    dfnp = df    # dfnp = df[(df.load!='primary [kWh/m2]')]
    dfnp = dfnp[(dfnp.model == 'xgb_lr0.1') &
                (dfnp.num_inputs == 8) &
                (dfnp.sampling_method == 'mipt_full')]
    dfnp['load'] = dfnp['load'].replace({'cool_load [kWh/m2]': 'Cooling load',
                                         'heat_load [kWh/m2]': 'Heating load',
                                         'light_load [kWh/m2]': 'Lighting load',
                                         'primary [kWh/m2]': 'Primary energy'})
    dfnp['setpoints'] = np.where(dfnp['heat_SP']==19,
                                 np.where(dfnp['cool_SP']==24,
                                          'hsp=19C,\ncsp=24C',
                                          'hsp=19C,\ncsp=26C'),
                                 np.where(dfnp['cool_SP']==24,
                                          'hsp=21C,\ncsp=24C',
                                          'hsp=21C,\ncsp=26C'))

    sns.set_theme(style="whitegrid", font_scale=1.6)
    g = sns.catplot(data = dfnp,
                    kind = 'box',
                    x = 'setpoints',
                    y = 'cvrmse',
                    col = 'load',
                    errorbar = None,
                    palette='Set1',        # alternatives: tab10, Set1
                    saturation = 0.75,
                    height = 10,
                    aspect = 0.8)
    g.despine(left = True)
    g.set_titles('{col_name}')
    g.set_axis_labels("", "Distribution of CV(RMSE)")

    # now prepare simulated results
    dfs = pd.read_csv('cv_qcd_simulated_results.csv')
    # dfs = dfs[dfs.load!='primary [kWh/m2]']
    dfs['setpoints'] = np.where(dfs['heat_SP']==19,
                                np.where(dfs['cool_SP']==24,
                                         'hsp=19C,\ncsp=24C',
                                         'hsp=19C,\ncsp=26C'),
                                np.where(dfs['cool_SP']==24,
                                         'hsp=21C,\ncsp=24C',
                                         'hsp=21C,\ncsp=26C'))

    # draw each small box plot as the inset in one of the large box plots
    axins = [inset_axes(ax, '30%', '30%', loc='upper right', borderpad=0) for ax in g.axes.ravel()]
    for ax in axins:
        ax.tick_params(labelbottom=False)
    sns.boxplot(data=dfs[dfs.load=='heat_load [kWh/m2]'],
                x='setpoints',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[0])
    sns.boxplot(data=dfs[dfs.load=='cool_load [kWh/m2]'],
                x='setpoints',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[1])
    sns.boxplot(data=dfs[dfs.load=='light_load [kWh/m2]'],
                x='setpoints',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[2])
    sns.boxplot(data=dfs[dfs.load=='primary [kWh/m2]'],
                x='setpoints',
                y='cv',
                showfliers=False,
                saturation = 0.75,
                ax=axins[3])

    g.savefig('fig_models/d5_boxplot_setpoints_renew.png')


def draw_boxplot_D6():
    df = pd.read_csv('cvrmse_xgb_predicted_special.csv')
    # df = df[df.load!='primary [kWh/m2]']
    # replace within the load column: cryptic
    df['load'] = df['load'].replace({'cool_load [kWh/m2]': 'Cooling load',
                                     'heat_load [kWh/m2]': 'Heating load',
                                     'light_load [kWh/m2]': 'Lighting load',
                                     'primary [kWh/m2]': 'Primary energy'})

    df_heat = df[df.load == 'Heating load']
    df_cool = df[df.load == 'Cooling load']
    df_light = df[df.load == 'Lighting load']
    df_primary = df[df.load == 'Primary energy']

    sns.set_theme(style="whitegrid", font_scale=1.6)

    gh = sns.catplot(data = df_heat,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.75)
    gh.despine(left = True)
    # gh.set_titles('{col_name}')
    gh.fig.suptitle('Heating load')
    gh.set_axis_labels("Sample size", "Distribution of CV(RMSE)")
    gh.savefig('fig_models/d6_boxplot_heat.png')

    gc = sns.catplot(data = df_cool,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.75)
    gc.despine(left = True)
    # gc.set_titles('{col_name}')
    gc.fig.suptitle('Cooling load')
    gc.set_axis_labels("Sample size", "Distribution of CV(RMSE)")
    gc.savefig('fig_models/d6_boxplot_cool.png')

    gl = sns.catplot(data = df_light,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.75)
    gl.despine(left = True)
    # gl.set_titles('{col_name}')
    gl.fig.suptitle('Lighting load')
    gl.set_axis_labels("Sample size", "Distribution of CV(RMSE)")
    gl.savefig('fig_models/d6_boxplot_light.png')

    gp = sns.catplot(data = df_primary,
                     kind = 'box',
                     x = 'model',
                     y = 'cvrmse',
                     # showfliers = False,
                     errorbar = None,
                     palette='Set1',        # alternatives: tab10, Set1
                     saturation = 0.75,
                     height = 10,
                     aspect = 0.75)
    gp.despine(left = True)
    # gh.set_titles('{col_name}')
    gp.fig.suptitle('Primary energy')
    gp.set_axis_labels("Sample size", "Distribution of CV(RMSE)")
    gp.savefig('fig_models/d6_boxplot_primary.png')


def draw_heatmap_D7():
    """
    Creates heat maps for heating, cooling and lighting loads
    for predictions of each of xgb12, xgb25, xgb50 and xgb100 models.
    """
    df_pred = pd.read_csv('starting_case_predictions.csv')
    df_sample = pd.read_csv('starting_case_sample_set.csv')

    for (model, sample_size) in [('xgb12', 12), ('xgb25', 25), ('xgb50', 50), ('xgb100', 100)]:
        draw_heatmap_for_model(model, df_pred, df_sample, sample_size)


def draw_heatmap_for_model(model, df_pred, df_sample, sample_size):
    fig, (axh, axc, axl, axp) = plt.subplots(nrows=4, ncols=1, sharex=True,
                                             figsize=(6,8),
                                             dpi=400, layout='constrained')
    fig.suptitle(f'Relative errors of {model} predictions')

    # heatmap swirls the coordinates around...
    df_sample['depth2'] = 50*(df_sample['depth'] + 0.01)
    df_sample['height2'] = 50*(0.5-df_sample['height'])

    df_pred['rel_heat_error'] = df_pred[f'pred_heat_{model}'] / df_pred['heat_load [kWh/m2]'] - 1.0
    dfh = pd.pivot_table(data=df_pred, index='height', columns='depth', values='rel_heat_error')
    dfh = dfh.reindex(index=dfh.index[::-1])
    sns.heatmap(data=dfh, vmin=-0.1, vmax=0.1,
                cmap=colorcet.CET_D1A, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, xticklabels=5,
                ax=axh)
    sns.scatterplot(data=df_sample[:sample_size], x='depth2', y='height2', ax=axh, s=8, marker='o', c='g')
    axh.set_title('Heating load', rotation='vertical', x=-0.17, y=0.075)

    df_pred['rel_cool_error'] = df_pred[f'pred_cool_{model}'] / df_pred['cool_load [kWh/m2]'] - 1.0
    dfc = pd.pivot_table(data=df_pred, index='height', columns='depth', values='rel_cool_error')
    dfc = dfc.reindex(index=dfc.index[::-1])
    sns.heatmap(data=dfc, vmin=-0.1, vmax=0.1,
                cmap=colorcet.CET_D1A, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, xticklabels=5,
                ax=axc)
    sns.scatterplot(data=df_sample[:sample_size], x='depth2', y='height2', ax=axc, s=8, marker='o', c='g')
    axc.set_title('Cooling load', rotation='vertical', x=-0.17, y=0.075)

    df_pred['rel_light_error'] = df_pred[f'pred_light_{model}'] / df_pred['light_load [kWh/m2]'] - 1.0
    dfl = pd.pivot_table(data=df_pred, index='height', columns='depth', values='rel_light_error')
    dfl = dfl.reindex(index=dfl.index[::-1])
    sns.heatmap(data=dfl, vmin=-0.1, vmax=0.1,
                cmap=colorcet.CET_D1A, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, xticklabels=5,
                ax=axl)
    sns.scatterplot(data=df_sample[:sample_size], x='depth2', y='height2', ax=axl, s=8, marker='o', c='g')
    axl.set_title('Lighting load', rotation='vertical', x=-0.17, y=0.075)

    df_pred['rel_primary_error'] = df_pred[f'pred_primary_{model}'] / df_pred['primary [kWh/m2]'] - 1.0
    dfp = pd.pivot_table(data=df_pred, index='height', columns='depth', values='rel_primary_error')
    dfp = dfp.reindex(index=dfp.index[::-1])
    sns.heatmap(data=dfp, vmin=-0.1, vmax=0.1,
                cmap=colorcet.CET_D1A, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, xticklabels=5,
                ax=axp)
    sns.scatterplot(data=df_sample[:sample_size], x='depth2', y='height2', ax=axp, s=8, marker='o', c='g')
    axp.set_title('Primary energy', rotation='vertical', x=-0.17, y=0.075)

    fig.savefig(f'fig_models/d7_heatmap_{model}.png')


def draw_visualizations_D8():
    """
    Uses vedo to draw 3d visualizations of predictions made by various xgb models,
    to illustrate the point that xgb models prefer to learn functions with smaller gradients.
    """
    print(f'loading predictions...')
    df = pd.read_csv('starting_case_predictions.csv')
    df = df.sort_values(['height', 'depth'])

    xsize = df['depth'].nunique()
    ysize = df['height'].nunique()

    X = df['depth'].to_numpy()
    Y = df['height'].to_numpy()
    Ybackwards = 0.5-Y         # reverse the Y-values in order to Y-mirror the lighting load diagram

    # we need to prepare visualizations for all 16 following columns:
    # pred_heat_xgb12, pred_heat_xgb25, pred_heat_xgb50, pred_heat_xgb100,
    # pred_cool_xgb12, pred_cool_xgb25, pred_cool_xgb50, pred_cool_xgb100,
    # pred_light_xgb12, pred_light_xgb25, pred_light_xgb50, pred_light_xgb100,
    # pred_primary_xgb12, pred_primary_xgb25, pred_primary_xgb50, pred_primary_xgb100
    print(f'translating prediction values...')
    models = ['xgb12', 'xgb25', 'xgb50', 'xgb100']
    # ZH = [(df[f'pred_heat_{model}'].to_numpy() - 30) / 30 for model in models]
    # ZC = [(df[f'pred_cool_{model}'].to_numpy() + 2.5) / 45 for model in models]
    # ZL = [df[f'pred_light_{model}'].to_numpy() for model in models]
    ZP = [(df[f'pred_primary_{model}'].to_numpy() - 190 ) / 100 for model in models]

    # mesh vertices
    # vertsH = [list(zip(X,Y,zh)) for zh in ZH]
    # vertsC = [list(zip(X,Y,zc)) for zc in ZC]
    # vertsL = [list(zip(X,Ybackwards,zl)) for zl in ZL]
    vertsP = [list(zip(X,Y,zp)) for zp in ZP]

    # mesh faces, i.e., triangles
    faces = [(xsize*j+i, xsize*j+i+1, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)] + \
            [(xsize*j+i, xsize*j+xsize+i, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)]

    # meshes themselves
    # draw_heating_visualizations_D8(ZH, vertsH, faces, models)
    # draw_cooling_visualizations_D8(ZC, vertsC, faces, models)
    # draw_lighting_visualizations_D8(ZL, vertsL, faces, models)
    draw_primary_visualizations_D8(ZP, vertsP, faces, models)


def draw_heating_visualizations_D8(ZH, vertsH, faces, models):
    print(f'preparing the heating load visualizations...')
    for i in range(len(ZH)):
        meshH = Mesh([vertsH[i], faces])
        meshH.pointdata['heating load'] = ZH[i]       # you must first associate numerical data to its points
        meshH.pointdata.select('heating load')        # and then make them "active"
        meshH.cmap(colorcet.CET_L3)     # ('terrain')

        isolH = meshH.isolines(n=27).color('w')
        isolH.lw(3)

        camH = dict(
            position=(-3, -2.25, 2.0),
            focal_point=(0.75, 0.25, 0.6),
            viewup=(0, 0, 1),
            distance=3.0,
            clipping_range=(1.0, 6.0),
        )
        lightH = Light(pos=(-1, 1, 3), focal_point=camH["focal_point"], c='w', intensity=1)

        pltH = Plotter(N=1, size=(1200,1000),
                       axes = dict(xtitle='depth (m)',
                                   xtitle_offset=0.175,
                                   xtitle_size=0.0165,
                                   xtitle_position=0.24,
                                   xlabel_size=0.012,
                                   xaxis_rotation=90,
                                   xygrid=True,
                                   ytitle='height (m)',
                                   ytitle_offset=0.025,
                                   ytitle_size=0.015,
                                   ytitle_rotation=(-90,0,90),
                                   ytitle_position=0.65,
                                   ylabel_size=0.012,
                                   ylabel_offset=0.85,
                                   yzgrid=False,
                                   yzgrid2=True,
                                   ztitle='heating load (kWh/m2)',
                                   ztitle_offset=-0.06,
                                   ztitle_position=1.05,
                                   ztitle_size=0.015,
                                   ztitle_rotation=(90,0,15),
                                   zlabel_size=0.012,
                                   zaxis_rotation=-105,
                                   zshift_along_x = 1,
                                   zrange=(0.4, 1.01),
                                   z_values_and_labels=[(i, f'{30*i+30:.2f}') for i in np.linspace(0.4, 1.0, 7)],
                                   zxgrid2=True,
                                   axes_linewidth=3,
                                   grid_linewidth=2,
                                   number_of_divisions=16,
                                   text_scale=1.8)).parallel_projection(value=True)
        pltH.show(meshH, isolH, lightH, camera=camH, interactive=False, zoom=2)
                                  # interactive=False when you know all the settings
        pltH.screenshot(f'fig_models/d8_pred_heat_{models[i]}.png')   # and uncomment this to save the view to the external file
        pltH.close()


def draw_cooling_visualizations_D8(ZC, vertsC, faces, models):
    print(f'preparing the cooling load visualizations...')
    for i in range(len(ZC)):
        meshC = Mesh([vertsC[i], faces])
        meshC.pointdata['cooling load'] = ZC[i]       # you must first associate numerical data to its points
        meshC.pointdata.select('cooling load')        # and then make them "active"
        meshC.cmap(colorcet.CET_L7)     # ('terrain')

        isolC = meshC.isolines(n=27).color('w')
        isolC.lw(3)

        camC = dict(
            position=(3.85577, -1.73075, 1.59065),
            focal_point=(0.86, 0.22, 0.925),  # 0.925060),
            viewup=(0, 0, 1),  # (-0.139573, 0.112697, 0.983778),
            distance=3.73636,
            clipping_range=(1.5, 6.11313),
        )
        lightC = Light(pos=(2, 0, 2), focal_point=camC["focal_point"], c='w', intensity=1)
        pltC = Plotter(N=1, size=(1250, 1000),
                       axes=dict(xtitle='depth',
                                 xtitle_offset=0.175,
                                 xtitle_size=0.0175,
                                 xlabel_size=0.012,
                                 xaxis_rotation=90,
                                 xygrid=True,
                                 ytitle='height',
                                 ytitle_offset=-0.115,
                                 ytitle_size=0.0175,
                                 ylabel_size=0.012,
                                 yshift_along_x=1.0,
                                 ylabel_offset=-2.5,
                                 yaxis_rotation=90,
                                 yzgrid=True,
                                 ztitle='cooling load (kWh/m2)',
                                 ztitle_offset=0.05,
                                 ztitle_size=0.0175,
                                 ztitle_position=1,
                                 zlabel_size=0.012,
                                 zaxis_rotation=45,
                                 zrange=(2/3, 13/9+0.001),
                                 z_values_and_labels=[(i, f'{45 * i - 2.5:.2f}') for i in np.linspace(6/9, 13/9, 8)],
                                 zxgrid2=True,
                                 axes_linewidth=3,
                                 grid_linewidth=2,
                                 number_of_divisions=16,
                                 text_scale=1.8)).parallel_projection(value=True)
        pltC.show(meshC, isolC, lightC, camera=camC, interactive=False, zoom=1.5)
        pltC.screenshot(f'fig_models/d8_pred_cool_{models[i]}.png')
        pltC.close()


def draw_lighting_visualizations_D8(ZL, vertsL, faces, models):
    print(f'preparing the lighting load visualizations...')
    for i in range(len(ZL)):
        meshL = Mesh([vertsL[i], faces])
        meshL.pointdata['lighting load'] = ZL[i]    # you must first associate numerical data to its points
        meshL.pointdata.select('lighting load')     # and then make them "active"
        meshL.cmap(colorcet.CET_L17)     # ('terrain')

        isolL = meshL.isolines(n=27).color('w')
        isolL.lw(3)

        camL = dict(
            position=(-0.25, -5, 12.3),
            focal_point=(0.725, -0.25, 11.075),
            viewup=(0, 0, 1),
            distance=5,
            clipping_range=(1.0, 7.0),
        )
        lightL = Light(pos=(-0.25, -5, 14), focal_point=camL["focal_point"], c='w', intensity=1.25)
        pltL = Plotter(N=1, size=(1600, 850),
                       axes=dict(xtitle='depth',
                                 xtitle_offset=0.2,
                                 xtitle_size=0.0175,
                                 xtitle_position=0.185,
                                 xlabel_size=0.012,
                                 xaxis_rotation=90,
                                 xygrid=True,
                                 ytitle='height',
                                 ytitle_offset=-0.03,
                                 ytitle_size=0.0175,
                                 ytitle_rotation=(0, 0, 90),
                                 ytitle_position=1.25,
                                 ylabel_size=0.012,
                                 ylabel_offset=0.75,
                                 ylabel_justify='center-right',
                                 y_values_and_labels=[(i, f'{(0.5 - i):.1f}') for i in np.linspace(0.1, 0.4, 4)],
                                 yzgrid=False,
                                 yzgrid2=True,
                                 ztitle='lighting load (kWh/m2)',
                                 ztitle_offset=-0.17,
                                 ztitle_size=0.0175,
                                 ztitle_position=1,
                                 zlabel_size=0.012,
                                 zlabel_offset=-0.75,
                                 zlabel_justify='center-left',
                                 zaxis_rotation=-33,
                                 zshift_along_x=1,
                                 zrange=(10.7, 11.4),
                                 z_values_and_labels=[(i, f'{i:.1f}') for i in np.linspace(10.7, 11.4, 8)],
                                 zxgrid=False,
                                 zxgrid2=True,
                                 axes_linewidth=3,
                                 grid_linewidth=2,
                                 number_of_divisions=17,
                                 text_scale=1.8)).parallel_projection(value=True)
        pltL.show(meshL, isolL, lightL, camera=camL, interactive=False, zoom=2.55)
        pltL.screenshot(f'fig_models/d8_pred_light_{models[i]}.png')
        pltL.close()


def draw_primary_visualizations_D8(ZP, vertsP, faces, models):
    print(f'preparing the primary energy visualizations...')
    for i in range(len(ZP)):
        meshP = Mesh([vertsP[i], faces])
        meshP.pointdata['primary energy'] = ZP[i]    # you must first associate numerical data to its points
        meshP.pointdata.select('primary energy')     # and then make them "active"
        meshP.cmap(colorcet.CET_L20)     # ('terrain')

        isolP = meshP.isolines(n=27).color('w')
        isolP.lw(3)

        camP = dict(
            position=(-4, -5, 3),
            focal_point=(0.35, -0.3, 0.95),
            viewup=(0, 0, 1),
            distance=5,
            clipping_range=(1.0, 7.0))
        lightP = Light(pos=(0.8, 0, 4.0), focal_point=camP["focal_point"], c='w', intensity=1)
        pltP = Plotter(N=1, size=(1700, 1000),
                       axes=dict(xtitle='depth',
                                 xtitle_offset=0.18,
                                 xtitle_size=0.0175,
                                 xtitle_position=0.16,
                                 xlabel_size=0.012,
                                 xaxis_rotation=90,
                                 xygrid=True,
                                 ytitle='height',
                                 ytitle_offset=0.03,
                                 ytitle_position=0.775,
                                 ytitle_size=0.0175,
                                 ytitle_rotation=(-90, 0, 90),
                                 ylabel_size=0.012,
                                 # yshift_along_x=1.0,
                                 # ylabel_offset=0.7,
                                 ylabel_justify='center-right',
                                 yaxis_rotation=0,
                                 yzgrid=False,
                                 yzgrid2=True,
                                 ztitle='primary energy (kWh/m2)',
                                 ztitle_offset=-0.12,
                                 ztitle_position=-0.155,
                                 ztitle_size=0.0175,
                                 ztitle_rotation=(0,44,0),
                                 zlabel_size=0.012,
                                 zlabel_justify='center-left',
                                 zlabel_offset=-0.6,
                                 zshift_along_x=1.0,
                                 zaxis_rotation=-60,
                                 zrange=(0.6, 1.001),
                                 z_values_and_labels=[(i, f'{190 + 100 * i:.2f}') for i in
                                                      np.linspace(0.6, 1.0, 9)],
                                 zxgrid2=True,
                                 axes_linewidth=3,
                                 grid_linewidth=2,
                                 number_of_divisions=16,
                                 text_scale=1.8)).parallel_projection(value=True)
        pltP.show(meshP, isolP, lightP, camera=camP, interactive=False, zoom=3.75)
        pltP.screenshot(f'fig_models/d8_pred_primary_{models[i]}.png')
        pltP.close()


if __name__=="__main__":
    # join_model_results()
    # reorganize_cvrmse_results()

    # draw_barplot_D1()
    # draw_boxplot_D1()
    # draw_barplot_D3()
    # draw_boxplot_D3()
    # draw_boxplot_D4()
    # compute_cv_qcd_simulated_results()
    # draw_boxplot_D5_climate_renew()
    # draw_boxplot_D5_obstacles_renew()
    # draw_boxplot_D5_orientation_renew()
    # draw_boxplot_D5_setpoints_renew()
    # reorganize_cvrmse_results_2()
    # draw_boxplot_D6()
    # draw_heatmap_D7()
    # draw_visualizations_D8()
