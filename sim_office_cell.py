import multiprocessing as mp
from eppy.modeleditor import IDF
from eppy.runner.run_functions import EnergyPlusRunError
import os
import shutil
import pandas as pd
import datetime
import numpy as np
import glob

from create_office_cell import create_office_cell_model

def generate_params():
    """
    Generator of all parameter tuples for m1overhang models that will be finally simulated.

    Currently selects a particular selection of parameters other than the overhang depth and height.
    Remove commented parts to general all parameter tuples.

    :return: a new (climate, obstacle, height, depth, orientation, heat_SP, cool_SP) tuple on each call
    """
    height_range = np.linspace(0.01, 0.49, 25).round(decimals=2)
    depth_range = np.linspace(0.0, 1.6, 81).round(decimals=2)

    for climate in [4]: # range(6):
        for obstacle in [0]: # range(5):
            for height in height_range:
                for depth in depth_range:
                    for orientation in [0.0]: # [0.0, 45.0, -45.0]:
                        for heat_SP in [21]: # [19, 21]:
                            for cool_SP in [24]: # [24, 26]:
                                yield (climate, obstacle, height, depth, orientation, heat_SP, cool_SP)


# at the moment, a global constant specifying whether to delete the output files
deleteOutputFiles = True

def simulate(params):
    # first create the idf object from supplied model parameters
    idf = create_office_cell_model(*params)

    # prepare the arguments for running Energyplus
    idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idf_version.extend([0] * (3 - len(idf_version)))
    idf_version_str = '-'.join([str(item) for item in idf_version])

    file_name = idf.idfname
    # remove the extension from the filename to serve as the prefix for output files
    prefix = os.path.basename(file_name)
    dot_pos = prefix.rindex('.')
    prefix = prefix[0:dot_pos]

    args = {
        'ep_version': idf_version_str,  # runIDFs needs the version number
        'output_directory': os.path.join(os.path.dirname(file_name), 'exec', prefix),
        'output_prefix': prefix,
        'output_suffix': 'D',
        'readvars': True,
        'expandobjects': True,
        'epmacro': True,
        'verbose': 'q'
    }

    # has this case been simulated already - maybe computer crashed in the meantime?
    saved_meter = os.path.join(os.path.dirname(file_name), 'results', f'{prefix}-meter.csv')
    if os.path.isfile(saved_meter):
        print(f'Meter {saved_meter} already exists - skipping simulation...')
        return saved_meter

    # otherwise, make sure the output directory is not there
    shutil.rmtree(args['output_directory'], ignore_errors=True)
    # create the output directory
    os.mkdir(args['output_directory'])

    try:
        # run Energyplus
        print(f'Starting EnergyPlus simulation for {idf.idfname}')
        idf.run(**args)

        # readvars was set to True, so ReadVarsESO collected the meters in a separate csv file
        # copy this cvs file to the results subfolder
        shutil.copyfile(os.path.join(args['output_directory'], f'{prefix}-meter.csv'),
                        saved_meter)

        # and then remove the output directory
        if deleteOutputFiles:
            shutil.rmtree(args['output_directory'], ignore_errors=True)

        # finally, let collect_results() know where to find the output meter file
        return saved_meter
    except EnergyPlusRunError as e:
        print('Simulation run failed: ' + str(e))


def collect_results(deleteMeterFiles=True):
    """
    Goes through all *-meter.csv files in the results folder, and
    collects data from them.

    :param deleteMeterFiles: whether to delete processed meter files (default=True)
    :return: The name of the file with collected results
    """
    # create a list of tuples for collecting the data
    data = []

    # get the list of meter files
    meter_files = glob.glob('results/*-meter.csv')

    # now go through the list of individual meter files
    for file_name in meter_files:
        if not file_name is None:
            # parse the file name to obtain the model parameters
            # its format is f'm1_cl_{climate}_ob_{obstacle}_h_{height}_d_{depth}_or_{orientation}_hsp_{heat_SP}_csp_{cool_SP}-meter.csv'
            parts = file_name.split('_')
            climate = int(parts[2])
            obstacle = int(parts[4])
            height = float(parts[6])
            depth = float(parts[8])
            orientation = float(parts[10])
            heat_SP = float(parts[12])
            # cool_SP has a trailing "-meter.csv", remove that first
            cool_SP = float(parts[14].removesuffix("-meter.csv"))

            try:
                # read the csv file to obtain heating, cooling, lighting energies
                tmp_df = pd.read_csv(file_name)
                heating = tmp_df.iloc[0, 1]
                cooling = tmp_df.iloc[0, 2]
                lighting = tmp_df.iloc[0, 3]

                # add that information to the list of tuples
                data.append((climate, obstacle, height, depth, orientation, heat_SP, cool_SP, heating, cooling, lighting))
            except:
                pass
                # find_unsimulated will catch all parameter tuples for which we did not reach data.append above...

            # delete the meter file if so requested
            if deleteMeterFiles:
                os.unlink(file_name)

    # create a dataframe from the collected data
    df = pd.DataFrame.from_records(data=data,
                                   columns=['climate', 'obstacle', 'height', 'depth',
                                            'orientation', 'heat_SP', 'cool_SP',
                                            'heating energy [J]', 'cooling energy [J]', 'lighting energy [J]'])
    # and save its contents to the final csv file
    timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    results_file = 'collected_results_'+timestamp+'.csv'
    # must specify the float_format to avoid 0.15 being saved as 0.150000000000002...
    df.to_csv(results_file, index=False, float_format='%g')
    return results_file


def find_unsimulated():
    """
    Goes through all collected_results_*.csv files in the current folder, and
    checks which param tuples have not been simulated yet.
    Should work much faster as it uses set difference() method of pandas' MultiIndex,
    instead of checking existence of parameters one-by-one.

    :return: The name of the file with unsimulated parameter tuples (in csv format)
    """

    # collect the results from all files in a single dataframe
    result_files = glob.glob('collected_results*.csv')
    sim_df = pd.concat(map(pd.read_csv, result_files), ignore_index=True)
    # drop the energy columns from the dataframe - we're interested in params only
    # plus, don't make a copy of the dataframe, but continue to work with this one (inplace=True)
    sim_df.drop(columns=['heating energy [J]', 'cooling energy [J]', 'lighting energy [J]'], inplace=True)

    # collect the parameters in a single dataframe
    param_list = []
    params = generate_params()      # restarts the generator by creating a new instance of it
    for p in params:
        (climate, obstacle, height, depth, orientation, heat_SP, cool_SP) = p
        param_list.append({'climate': climate,
                            'obstacle': obstacle,
                            'height': height,
                            'depth': depth,
                            'orientation': orientation,
                            'heat_SP': heat_SP,
                            'cool_SP': cool_SP})
    param_df = pd.DataFrame.from_records(data=param_list)

    # convert dataframes to multiindices to find their set difference
    sim_mi = pd.MultiIndex.from_frame(sim_df)
    param_mi = pd.MultiIndex.from_frame(param_df)
    unsim_mi = param_mi.difference(sim_mi, sort=False)

    # back to dataframe and then out to an external file
    unsim_df = unsim_mi.to_frame(index=False)
    timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    unsim_file = 'unsimulated_params_'+timestamp+'.csv'
    # do not forget to specify float_format to avoid 0.15 being saved as 0.150000000002
    unsim_df.to_csv(unsim_file, index=False, float_format='%g')
    return unsim_file


def run_from_param_generator():
    """
    (Initial) run of simulations using the parameter generator.
    """
    params = generate_params()
    with mp.Pool(max(mp.cpu_count(), 1)) as pool:
        pool.map(simulate, params)


def run_from_param_file(unsim_file):
    df = pd.read_csv(unsim_file)
    params = df.itertuples(index=False, name=None)
    with mp.Pool(max(mp.cpu_count(), 1)) as pool:
        pool.map(simulate, params)


if __name__ == '__main__':
    idd_file = 'Energy+.idd'
    try:
        IDF.setiddname(idd_file)
    except modeleditor.IDDAlreadySetError as e:
        pass

    mp.freeze_support()     # may have effect in Windows...

    ################################################
    # UNCOMMENT THE APPROPRIATE PIECE OF CODE BELOW
    ################################################

    # If you're starting simulations for the first time:
    print(f'Running simulations from parameter generator...')
    run_from_param_generator()

    # Collecting results
    print(f'Collecting results...')
    collect_results()

    # Are there any parameters for which simulations were not run?
    print(f'Collecting unsimulated parameter tuples...')
    unsim_file = find_unsimulated()

    # Now start the simulations only for those that were unsimulated
    # print(f'Running simulations from parameter file...')
    # unsim_file = 'unsimulated_params_23_03_20_18_04_04.csv'
    # run_from_param_file(unsim_file)
