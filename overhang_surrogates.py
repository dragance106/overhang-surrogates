"""
This file provides more general implementations of Python methods used to
create surrogate models for various loads of an office cell with attached overhang
with respect to overhang depth and height. Besides originally intended purpose,
these methods allow interested researchers to use them independently in order to:
- produce samples with mc-intersite-proj-th, the Monte Carlo sampling method
  described in K. Crombecq, E. Laermans, T. Dhaene, Efficient space-filling and
  non-collapsing sequential design strategies for simulation-based modeling,
  Eur J Oper Res 214 (2011), 683--696.
- attach overhang with given depth, height, slope and distance
  from the given window in the given EnergyPlus input data file (idf)
- add obstacles at given distance from the given window in the given idf
- simulate given idf with the attached overhang according to sampled overhang parameters,
  which returns a pandas dataframe with overhang parameters and simulation results
- train XGBoost surrogate models with k-fold cross validation
  for the given pandas dataframe
- prepare three-dimensional diagrams of selected numerical values with vedo.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from eppy import modeleditor
from eppy.modeleditor import IDF
from eppy.runner.run_functions import EnergyPlusRunError
from vedo import *
import colorcet
import multiprocessing as mp
import math
from itertools import combinations
from functools import reduce
import os
import shutil
import unicodedata
import re


########################################
# mc-intersite-proj-th (MIPT) sampling #
########################################
def proj_dist(candidates, sampled):
    """
    Computes projected distance from each candidate point to the set of already sampled points.
    For the candidate point c, its projected distance to sampled points is
    min_j min_k |c_k - sampled_{j,k}|

    :param candidates: the candidates for the new sample point
    :param sampled:    already selected sample points
    :return:           the sequence of projected distances of candidate points
    """
    prd = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        prd[i] = np.amin(np.abs(c-sampled))
    return prd


def inter_dist(candidates, sampled):
    """
    Computes intersite distance from each candidate point to the set of already sampled points.
    For the candidate point c, its intersite distance to sampled points is
    min_j sqrt(sum_k (c_k - sampled_{j,k})^2)

    :param candidates: the candidates for the new sample point
    :param sampled:    already selected sample points
    :return:           the sequence of intersite distances of candidate points
    """
    ind = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        ind[i] = np.amin(np.sqrt(np.sum((c-sampled)**2, axis=1)))
    return ind


def proj_quality(samples):
    min_proj = np.amin(np.abs(samples[0]-samples[1]))

    pairs = combinations(samples, 2)
    for (p, q) in pairs:
        proj = np.amin(np.abs(p-q))
        if proj < min_proj:
            min_proj = proj

    return min_proj


def inter_quality(samples):
    min_inter = np.sqrt(np.sum((samples[0]-samples[1])**2))

    pairs = combinations(samples, 2)
    for (p, q) in pairs:
        inter = np.sqrt(np.sum((p-q)**2))
        if inter < min_inter:
            min_inter = inter

    return min_inter


def MIPT(n, dim=2, alpha=0.5, k=100, negligible=1e-6):
    """
    Implementation of Crombecq's mc-intersite-proj-th sampling scheme,
    in which the new candidate points are generated only within the allowed intervals,
    obtained after subtracting from [0,1]^dim
    the hypercubes covering the minimum projected distance around the already selected sample points.

    :param n:     the number of sample points to be generated
    :param dim:   the dimension of the design space
    :param alpha: the tolerance parameter for the minimum projected distance:
                  each candidate point with projected distance smaller than alpha/n is discarded
    :param k:     the number of candidate points to be generated in the i-th iteration
                  (after i-1 points have already been generated)
                  will be equal to k*i
    :param negligible:   the value considered negligible when mutually comparing
                         boundaries of different intervals

    :return:      the sequence of n sampled points from [0,1]^dim.
    """

    # placeholder for the sampled points
    sample = np.zeros((n, dim))

    # the first point is just randomly generated
    rng = np.random.default_rng()
    sample[0] = rng.random((dim,))

    for s in range(1, n):
        # minimum allowed projected distance
        dmin = alpha/(s+1)

        # placeholder for the candidates
        candidates = np.zeros((k*s, dim))

        # for each coordinate x
        for x in range(dim):
            # determine the union of disjoint intervals left after removing from [0,1]
            # the intervals [sample[j,x]-dmin, sample[j,x]+dmin] for j=0,...,i-1
            start_intervals = [(0,1)]

            for j in range(s):
                # subtract [sample[j,x]-dmin, sample[j,x]+dmin] from each interval in intervals
                l2 = sample[j,x] - dmin
                u2 = sample[j,x] + dmin

                end_intervals = []
                for (l1, u1) in start_intervals:
                    if u2<l1+negligible:
                        end_intervals.append((l1,u1))
                    elif u1<l2+negligible:
                        end_intervals.append((l1,u1))
                    elif l2<l1+negligible and l1<u2+negligible and u2<u1+negligible:
                        end_intervals.append((u2,u1))
                    elif l1<l2+negligible and l2<u1+negligible and u1<u2+negligible:
                        end_intervals.append((l1,l2))
                    elif l1<l2+negligible and u2<u1+negligible:
                        end_intervals.append((l1,l2))
                        end_intervals.append((u2,u1))
                    else:
                        pass

                # now substitute end_intervals for start_intervals, and repeat
                start_intervals = end_intervals

            # after this loop finishes we have the requested union of allowed intervals,
            # so we want to generate k*i random values within them
            # to serve as the x-th coordinate for the set of candidates
            cum_length = np.zeros((len(start_intervals),))

            (l, u) = start_intervals[0]
            cum_length[0] = u-l

            # if len(start_intervals)>1:
            for i in range(1, len(start_intervals)):
                (l, u) = start_intervals[i]
                cum_length[i] = cum_length[i-1] + u-l

            total_length = cum_length[len(start_intervals)-1]

            # generate k*s random values within [0,1] and rescale them to total_length
            coords = total_length * rng.random((k*s,))

            # distribute them appropriately to the allowed intervals
            for j in range(k*s):
                for i in range(len(start_intervals)):
                    if coords[j] < cum_length[i] + 1e-8:   # just so that we do not miss total_length
                        break
                if i==0:
                    coords[j] = start_intervals[i][0] + coords[j]
                else:
                    coords[j] = start_intervals[i][0] + (coords[j]-cum_length[i-1])

            # assign final coordinates to the set of candidates
            candidates[:,x] = coords

        # candidates with proper projected distance from the existing sample points are now selected,
        # so proceed to compute their intersite distance to the existing sample points
        # and add the best candidate to the sample
        ind = inter_dist(candidates, sample[:s])
        sample[s] = candidates[np.argmax(ind)]

    # n points have been now sampled
    return sample


def MIPT_extend(current, n, alpha=0.5, k=100, negligible=1e-6):
    """
    Implementation of Crombecq's mc-intersite-proj-th sampling scheme
    that extends the current sample of points from the hypercube [0,1]^dim.
    New candidate sample points are generated only within the allowed intervals,
    obtained after subtracting from [0,1]^dim
    the hypercubes covering the minimum projected distance around the already selected sample points.

    :param current:     existing sample points from [0,1]^dim,
                        given as a numpy array, e.g.,
                        np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    :param n:     the number of new points to be added to the sample
    :param alpha: the tolerance parameter for the minimum projected distance:
                  each candidate point with projected distance smaller than alpha/n is discarded
    :param k:     the number of candidate points to be generated in the i-th iteration
                  (after i-1 points have already been generated)
                  will be equal to k*i
    :param negligible:   the value considered negligible when mutually comparing
                         boundaries of different intervals

    :return:      the current sample together with additional n points from [0,1]^dim,
                  where dim is the dimension of points in the current sample.
    """

    # the 0-th dimension will be the number of points in the current sample
    current_size = current.shape[0]
    dim = current.shape[1]

    # placeholder for the sampled points
    sample = np.zeros((current_size + n, dim))

    # copy existing sample points
    sample[0:current_size] = current

    # random number generator always comes handy
    rng = np.random.default_rng()

    for s in range(current_size, current_size + n):
        # minimum allowed projected distance
        dmin = alpha/(s+1)

        # placeholder for the candidates
        candidates = np.zeros((k*s, dim))

        # for each coordinate x
        for x in range(dim):
            # determine the union of disjoint intervals left after removing from [0,1]
            # the intervals [sample[j,x]-dmin, sample[j,x]+dmin] for j=0,...,i-1
            start_intervals = [(0,1)]

            for j in range(s):
                # subtract [sample[j,x]-dmin, sample[j,x]+dmin] from each interval in intervals
                l2 = sample[j,x] - dmin
                u2 = sample[j,x] + dmin

                end_intervals = []
                for (l1, u1) in start_intervals:
                    if u2<l1+negligible:
                        end_intervals.append((l1,u1))
                    elif u1<l2+negligible:
                        end_intervals.append((l1,u1))
                    elif l2<l1+negligible and l1<u2+negligible and u2<u1+negligible:
                        end_intervals.append((u2,u1))
                    elif l1<l2+negligible and l2<u1+negligible and u1<u2+negligible:
                        end_intervals.append((l1,l2))
                    elif l1<l2+negligible and u2<u1+negligible:
                        end_intervals.append((l1,l2))
                        end_intervals.append((u2,u1))
                    else:
                        pass

                # now substitute end_intervals for start_intervals, and repeat
                start_intervals = end_intervals

            # after this loop finishes we have the requested union of allowed intervals,
            # so we want to generate k*i random values within them
            # to serve as the x-th coordinate for the set of candidates
            cum_length = np.zeros((len(start_intervals),))

            (l, u) = start_intervals[0]
            cum_length[0] = u-l

            # if len(start_intervals)>1:
            for i in range(1, len(start_intervals)):
                (l, u) = start_intervals[i]
                cum_length[i] = cum_length[i-1] + u-l

            total_length = cum_length[len(start_intervals)-1]

            # generate k*s random values within [0,1] and rescale them to total_length
            coords = total_length * rng.random((k*s,))

            # distribute them appropriately to the allowed intervals
            for j in range(k*s):
                for i in range(len(start_intervals)):
                    if coords[j] < cum_length[i] + 1e-8:   # just so that we do not miss total_length
                        break
                if i==0:
                    coords[j] = start_intervals[i][0] + coords[j]
                else:
                    coords[j] = start_intervals[i][0] + (coords[j]-cum_length[i-1])

            # assign final coordinates to the set of candidates
            candidates[:,x] = coords

        # candidates with proper projected distance from the existing sample points are now selected,
        # so proceed to compute their intersite distance to the existing sample points
        # and add the best candidate to the sample
        ind = inter_dist(candidates, sample[:s])
        sample[s] = candidates[np.argmax(ind)]

    # n new points have been sampled now,
    # existing sample points are already at the beginning of the current sample
    return sample


def hypercube_to_indices(points, lengths):
    """
    Auxiliary method to convert a numpy array of points from the hypercube [0,1]^dim
    into indices of arrays with given lengths.

    :param points:   numpy array of points from the hypercube [0,1]^dim
    :param lengths:  lengths of arrays for which we need indices

    :return:    numpy array of indices from arrays with given length
    """
    indices = np.zeros(points.shape, dtype=int)
    for p in range(points.shape[0]):
        for i in range(points.shape[1]):
            indices[p, i] = math.floor(points[p, i] * lengths[i])
    return indices


def indices_to_hypercube(indices, lengths):
    """
    Auxiliary method to convert a numpy array of indices from arrays with given lengths
    into an array of points from the hypercube [0,1]^dim.
    For each dimension i,
    [0,1] is divided into lengths[i] intervals and
    any given index 0,...,lengths[i]-1 is mapped into the center of the corresponding interval.

    :param indices:   numpy array of indices from arrays with given lengths
    :param lengths:   lengths of arrays

    :return:    numpy array of points corresponding to given indices
                (centered within corresponding intervals)
    """
    points = np.zeros(indices.shape)
    for p in range(indices.shape[0]):
        for i in range(indices.shape[1]):
            points[p, i] = (indices[p, i] + 0.5) / lengths[i]
    return points


#################################################
# attaching overhang to a window in an idf file #
#################################################
def add_overhang(idf, window_name, height, depth, width):
    """
    This method adds an overhang to <idf>
    that is placed <height> meters above the <window_name>,
    is <depth> meters deep and <width> meters wide.
    The overhang starts from the wall on which the <window_name> resides.
    The <window_name> and the overhang are vertically centrally aligned.

    It is assumed that <window_name> belongs to an exterior vertical wall, and
    that <window_name> is a rectangular window whose all sides are either horizontal or vertical.

    IMPORTANT NOTE:
    EnergyPlus will report errors if any two vertices are placed less than 0.01m apart.
    Height is thus set to at least 0.01m by taking min(height, 0.01).
    Depth, if less than 0.01, will skip generating the overhang.

    :return: The idf object with attached overhang.
    """
    # get the list of all windows
    windows = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    # find those with the given name
    windows_named = [w for w in windows if w.Name == window_name]
    # find the (first) one with the given name, if it exists
    if len(windows_named) == 0:
        # no such window
        raise Exception(f'There is no window named {window_name} in the idf: {idf.idfname}')
    else:
        # take the first (and usually the only) window with that name
        window = windows_named[0]

    # attach an overhang to the "Office_Cell_Wall_South" wall if depth is at least 0.01m
    if depth>=0.01:
        # height should be at least 0.01m to avoid EnergyPlus reporting errors about vertex coincidences
        height = max(height, 0.01)

        overhang = idf.newidfobject('SHADING:ZONE:DETAILED')

        overhang.Name = 'Overhang_Above_' + window_name
        overhang.Base_Surface_Name = window.Building_Surface_Name
        overhang.Transmittance_Schedule_Name = ''
        overhang.Number_of_Vertices = 4

        # compute outward facing normal for the window and overhang direction vector
        a = [window.Vertex_2_Xcoordinate - window.Vertex_1_Xcoordinate,
             window.Vertex_2_Ycoordinate - window.Vertex_1_Ycoordinate,
             window.Vertex_2_Zcoordinate - window.Vertex_1_Zcoordinate]
        b = [window.Vertex_3_Xcoordinate - window.Vertex_2_Xcoordinate,
             window.Vertex_3_Ycoordinate - window.Vertex_2_Ycoordinate,
             window.Vertex_3_Zcoordinate - window.Vertex_2_Zcoordinate]

        out_normal = np.cross(a, b)
        oh_dir = np.cross(out_normal, [0, 0, 1])

        out_normal_length = math.sqrt(out_normal[0]**2 + out_normal[1]**2 + out_normal[2]**2)
        oh_dir_length = math.sqrt(oh_dir[0]**2 + oh_dir[1]**2 + oh_dir[2]**2)

        # out_normal and oh_dir are now unit vectors:
        out_normal /= out_normal_length
        oh_dir /= oh_dir_length

        # compute overhang vertices
        zmax = max(window.Vertex_1_Zcoordinate, window.Vertex_2_Zcoordinate,
                   window.Vertex_3_Zcoordinate, window.Vertex_4_Zcoordinate)
        xmin = min(window.Vertex_1_Xcoordinate, window.Vertex_2_Xcoordinate,
                   window.Vertex_3_Xcoordinate, window.Vertex_4_Xcoordinate)
        xmax = max(window.Vertex_1_Xcoordinate, window.Vertex_2_Xcoordinate,
                   window.Vertex_3_Xcoordinate, window.Vertex_4_Xcoordinate)
        ymin = min(window.Vertex_1_Ycoordinate, window.Vertex_2_Ycoordinate,
                   window.Vertex_3_Ycoordinate, window.Vertex_4_Ycoordinate)
        ymax = max(window.Vertex_1_Ycoordinate, window.Vertex_2_Ycoordinate,
                   window.Vertex_3_Ycoordinate, window.Vertex_4_Ycoordinate)

        center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, zmax+height])
        v1 = center + width*oh_dir/2
        v2 = center + width*oh_dir/2 + depth*out_normal
        v3 = center - width*oh_dir/2 + depth*out_normal
        v4 = center - width*oh_dir/2

        overhang.Vertex_1_Xcoordinate = v1[0]
        overhang.Vertex_1_Ycoordinate = v1[1]
        overhang.Vertex_1_Zcoordinate = v1[2]

        overhang.Vertex_2_Xcoordinate = v2[0]
        overhang.Vertex_2_Ycoordinate = v2[1]
        overhang.Vertex_2_Zcoordinate = v2[2]

        overhang.Vertex_3_Xcoordinate = v3[0]
        overhang.Vertex_3_Ycoordinate = v3[1]
        overhang.Vertex_3_Zcoordinate = v3[2]

        overhang.Vertex_4_Xcoordinate = v4[0]
        overhang.Vertex_4_Ycoordinate = v4[1]
        overhang.Vertex_4_Zcoordinate = v4[2]

        reflectance = idf.newidfobject('SHADINGPROPERTY:REFLECTANCE')
        reflectance.Shading_Surface_Name = overhang.Name
        reflectance.Diffuse_Solar_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
        reflectance.Diffuse_Visible_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
        reflectance.Fraction_of_Shading_Surface_That_Is_Glazed = 0
        reflectance.Glazing_Construction_Name = ''

    if idf.idfname.endswith('.idf'):
        idf.idfname = idf.idfname[:-4]
    idf.idfname = idf.idfname + f'_window={window_name}_h={height}_d={depth}_w={width}.idf'

    return idf


#######################################################################
# simulating an idf file with attached overhang with given parameters #
#######################################################################
def run_single_simulation(params):
    # unpack the parameters
    idd_filename, idf_filename, epw_filename, window_name, h, d, width, delete_output_files = params

    # set up the idd file and create the starting idf object
    try:
        IDF.setiddname(idd_filename)
    except modeleditor.IDDAlreadySetError as e:
        pass
    idf = IDF(idf_filename, epw_filename)

    # add the sampled overhang
    idf = add_overhang(idf, window_name, h, d, width)

    # prepare the arguments for running Energyplus
    idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idf_version.extend([0] * (3 - len(idf_version)))
    idf_version_str = '-'.join([str(item) for item in idf_version])

    file_name = idf.idfname
    prefix = os.path.basename(file_name)
    dot_pos = prefix.rindex('.')
    prefix = prefix[0:dot_pos]

    args = {
        'ep_version': idf_version_str,  # runIDFs needs the version number
        'output_directory': os.path.join(os.path.dirname(file_name), prefix),
        'output_prefix': prefix,
        'output_suffix': 'D',
        'readvars': True,
        'expandobjects': True,
        'epmacro': True,
        'verbose': 'q'
    }

    # make sure the output directory is not there
    shutil.rmtree(args['output_directory'], ignore_errors=True)
    # create the output directory
    os.mkdir(args['output_directory'])

    try:
        # run Energyplus
        print(f'Starting EnergyPlus simulation for {idf.idfname}')
        idf.run(**args)

        # read the csv file to obtain values of meters
        # it should contain a single line of values
        df_meters = pd.read_csv(os.path.join(args['output_directory'], f'{prefix}-meter.csv'))
        # add info on overhang height and depth
        df_meters['height'] = [h]
        df_meters['depth'] = [d]
    except EnergyPlusRunError as e:
        print('Simulation run failed: ' + str(e))
    except FileNotFoundError as e:
        print('CSV file with meters cannot be read: ' + str(e))
    except:
        pass

    # do not forget to remove the output directory
    if delete_output_files:
        shutil.rmtree(args['output_directory'], ignore_errors=True)

    # return the pandas dataframe with meters from this simulation
    # they will be merged after pool.map completes
    return df_meters


def sample_and_simulate_overhangs(idf_filename, epw_filename,
                                  window_name, depth_range, height_range, width,
                                  sample_size,
                                  idd_filename='Energy+.idd', delete_output_files=True):
    """
    This method makes a sample of overhang depths and heights of size <sample_size>,
    with the values of depths and heights sampled from
    the arrays <depth_range> and <height_range>, respectively.
    Afterwards, for each sampled overhang depth/height pair,
    it calls run_single_simulation through multiprocessing map method,
    which creates the IDF object, adds an overhang to it, simulates it
    and returns the values of the meters that are defined within <idf_filename> (!!!)
    using commands such as
        Output:Meter,DistrictHeating:Facility,RunPeriod;
    This method then combines all these values into a single pandas dataframe,
    which is then returned back to the caller.

    NOTE: This method reports spurious errors when run from PyCharm.
          Run it from the shell instead!

    :param idf_filename:   path to the idf file
    :param epw_filename:   path to the climate file corresponding to the idf file
    :param window_name:    name of the window above which overhangs should be placed
    :param depth_range:    array of possible overhang depths (in meters) from which to take the sample.
                           for example, if you have minimum and maximum possible depth,
                           and you know how many values you want in between, then you can use
                           depth_range = np.linspace(min_depth, max_depth, number_of_values).round(decimals=2)
                           round(decimals=2) (or decimals=3) is needed for nicer printout in pandas dataframe
                           (it's easier to look at 1.67 instead of 1.6666666666667)
    :param height_range:   array of possible overhang heights (in meters) from which to take the sample
                           similar suggestion stands for
                           height_range = np.linspace(min_height, max_height, number_of_values).round(decimals=2)
    :param width:          overhang width in meters, fixed value
    :param sample_size:    size of the sample with overhang depths and heights
    :param meters:         list of EnergyPlus loads which should be collected after simulations
    :param idd_filename:   path to the EnergyPlus idd file,
                           default 'Energy+.idd' means that you have its copy in the local directory
    :param delete_output_files:   whether to delete files left after EnergyPlus simulations,
                                  set to False if you need to inspect them afterwards

    :return:    pandas dataframe with columns containing sampled depths and heights and
                the metered values returned by EnergyPlus simulations
    """

    # make a sample of overhang heights and depths
    points = MIPT(sample_size)
    indices = hypercube_to_indices(points, [len(height_range), len(depth_range)])
    sample = [(height_range[i], depth_range[j]) for [i, j] in indices]

    # turn the sample into a parameter generator for multiprocessing
    params = ((idd_filename, idf_filename, epw_filename,
               window_name, h, d, width, delete_output_files) for (h, d) in sample)

    # run in parallel EnergyPlus simulations for each
    mp.freeze_support()     # may have effect with multiprocessing in Windows
    with mp.Pool(max(mp.cpu_count(), 1)) as pool:
        results = pool.map(run_single_simulation, params)

    # now concatenate all small dataframes from the results list
    df = pd.concat(results, ignore_index=True)
    # and reorganise the columns a bit
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[1:-2]   # first come height and depth, then the rest,
                                    # but skipping the superfluous Date/Time column
    df = df[cols]
    return df


######################################################################################
# training XGBoost model with k-fold cross validation for the given pandas dataframe #
######################################################################################
def train_model_ensembles(df, input_cols, meter_cols, num_folds=5, learning_rate=0.1, early_stopping_rounds=10):
    """
    For the dataframe <df>, the columns in <input_cols> are treated as inputs on which
    to train a separate ensemble of XGBoost models for each column from <meter_cols>.
    The method uses cross validation with <num_folds> folds,
    so that for each output column each ensemble is a list of <num_folds> XGBoost models.

    :return: Returns a dictionary of XGBoost ensembles,
             with each column from meter_cols as a key for the corresponding ensemble.
    """
    rng = np.random.default_rng()           # default random number generator
    rows = df.shape[0]
    rnd_indices = rng.permutation(rows)     # random permutation of row indices

    # split the dataframe randomly into <num_folds> folds
    # in cross validation, each fold serves as the validation set for one XGBoost model,
    # while the remaining folds serve as the training set for that XGBoost model
    train_folds = [df.iloc[np.concatenate((rnd_indices[0:rows*i//num_folds],
                                           rnd_indices[rows*(i+1)//num_folds:rows]))]
                   for i in range(num_folds)]
    test_folds = [df.iloc[rnd_indices[rows*i//num_folds:rows*(i+1)//num_folds]]
                  for i in range(num_folds)]

    ensembles = {}
    for meter in meter_cols:
        # train a new ensemble for each meter
        xgb_ensemble = []

        for i in range(num_folds):
            x_train = train_folds[i][input_cols]
            y_train = train_folds[i][meter]
            x_test = test_folds[i][input_cols]
            y_test = test_folds[i][meter]

            xgb_model = xgb.XGBRegressor(early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate)
            xgb_model.fit(x_train, y_train,
                          eval_set=[(x_test, y_test)],
                          verbose=False)

            xgb_ensemble.append(xgb_model)

        # add the trained ensemble to the list of all models
        ensembles[meter] = xgb_ensemble

    return ensembles


def predict_meters(ensembles, input_ranges):
    """
    Models is a dictionary,
    where each key represents a meter, and
    the value represents an ensemble of XGBoost models trained on the simulated sample.
    Predictions of an ensemble are obtained as the average of predictions of all models in it.
    The method makes predictions for each meter, and
    returns the dataframe consisting of predictions for each meter
    over all tuples of values from <input_ranges>

    :param ensembles:      dictionary with meters as key, and values as corresponding XGBoost ensembles
    :param input_ranges:   dictionary of ranges of input values
                           for which the models from the ensemble should make predictions.
                           For example,
                           {'depth': np.linspace(0.1, 1.0, 10).round(decimals=1),
                            'height': np.linspace(0.1, 0.5, 5).round(decimals=1)}.
                           This will internally be translated into the Cartesian product of these two arrays
                           that contains 50 input depth/height pairs for which the models will make predictions.

    :return:   dataframe with ensemble predictions for each meter and
               each element from the Cartesian product of given ranges.
    """
    # create a data frame for each input range
    input_frames = [pd.DataFrame(value, columns=[key]) for key, value in input_ranges.items()]
    # create the Cartesian product of all input ranges
    df = reduce(lambda left, right: pd.merge(left, right, how='cross'), input_frames)

    # make predictions for each ensemble
    for meter in ensembles:
        ensemble = ensembles[meter]
        sum_preds = np.zeros(df.shape[0])
        for model in ensemble:
            sum_preds += model.predict(df[list(input_ranges.keys())])
        sum_preds /= len(ensemble)

        # add predicted values as a new column to the dataframe
        df[meter] = sum_preds

    return df


#######################################
# preparing vedo diagrams for 3D data #
#######################################
def make_3d_diagram(df, xcol, ycol, zcol,
                    minxvalue=None, maxxvalue=None, numxticks=10,
                    minyvalue=None, maxyvalue=None, numyticks=10,
                    minzvalue=None, maxzvalue=None, numzticks=10,
                    filename=None,
                    light=None,
                    camera=(-15, 25),
                    aspect_ratio=(1, 1, 1),
                    palette=colorcet.CET_L20,
                    distance_multiplier=4,
                    num_isolines=25,
                    padding=0.001,
                    interactive=False,
                    diagram_size='auto',
                    **kwargs):
    """
    This method creates a three-dimensional diagram for data contained
    in the xcol, ycol and zcol columns of the dataframe df.
    It is assumed that the dataframe contains a single row for each pair
    from the Cartesian product of unique values that appear in xcol and ycol.
    The final diagram is saved to the external <filename>,
    although it is possible to set interactive=True in plt.show() call below
    to show the diagram in vedo's interactive mode.
    vedo's Plotter object has a myriad options for setting various parts of the diagram.
    Remaining **kwargs keyword parameters are forwarded to this object during its construction.
    Nevertheless, if you need more special formatting,
    it is advised that you copy the code below to another file and
    modify it accordingly.

    NOTE 1: vedo does not like to be run interactively from PyCharm,
            so the interactive version should be run from shell instead.
            Non-interactive version works like a charm from PyCharm...

    :param df:   pandas dataframe containing the numerical data for visualisation
    :param xcol: the column of df containing x-coordinates
    :param ycol: the column of df containing y-coordinates
    :param zcol: the column of df containing z-coordinates
                 it is assumed that each pair of unique values from xcol and ycol
                 appears in a unique row of df
    :param minxvalue: value for the smallest label on x-axis
                      if None, the minimum value from df[xcol] will be used
    :param maxxvalue: value for the largest label on x-axis
                      if None, the maximum value from df[xcol] will be used
    :param numxticks: number of labels on x-axis (including both smallest and largest)
    :param minyvalue: value for the smallest label on y-axis
                      if None, the minimum value from df[ycol] will be used
    :param maxyvalue: value for the largest label on y-axis
                      if None, the maximum value from df[ycol] will be used
    :param numyticks: number of labels on y-axis (including both smallest and largest)
    :param minzvalue: value for the smallest label on z-axis
                      if None, the minimum value from df[zcol] will be used
    :param maxzvalue: value for the largest label on z-axis
                      if None, the maximum value from df[zcol] will be used
    :param numzticks: number of labels on z-axis (including both smallest and largest)
    :param filename: filename under which to save the final diagram
                     if left as None, it will be saved in the current directory
                     with the filename "diagram_<xcol>_<ycol>_<zcol>.png"
                     (if a file with that filename already exists, it will be overwritten)
    :param light: vedo Light object
                  if left as None, it will be constructed internally
                  to shine from above upon the diagram center
    :param camera: either vedo camera dictionary
                   or a tuple containing azimuth and altitude of camera (in degrees)
                   altitude is the angle between xy-plane and camera position
                   azimuth is the angle in xy-plane between camera position and south direction (negative y)
    :param aspect_ratio: the mutual ratio of the width (x), depth (y) and height (z) of the diagram
    :param palette: color map to use
                    see https://colorcet.holoviz.org/user_guide/Continuous.html
                    for the complete set of perceptually uniform continuous color maps
                    (personal favorites are colorcet.CET_L3, CET_L7, CET_L17 and CET_L20)
    :param distance_multiplier: for camera given as (azimuth, altitude),
                    this parameter shows its distance from the diagram center
                    in multiples of halves of the diagonal of the diagram bounding box
    :param num_isolines: number of isolines to put in the diagram
    :param padding: extra space added to x-, y- and z-ranges to ensure
                    showing ticks for the extremal values
                    (padding is added to normalised aspect_ratio)
    :param interactive: whether to run vedo interactively (True) or not (False)
    :param diagram_size: size parameter for vedo Plotter object
    :param kwargs: dictionary of parameter keywords that are forwarded
                   to the axes of vedo Plotter object

    :return: None
    """
    df = df.sort_values([ycol, xcol])

    # make the smallest aspect_ratio equal to 1
    min_ar = min(aspect_ratio)
    if abs(min_ar) == 0:
        raise ValueError('Minimum aspect ratio entry must not be equal to zero.')
    aspect_ratio = [a/min_ar for a in aspect_ratio]

    # coordinate arrays, mapped into the aspect ratio bounding box
    X = df[xcol].to_numpy()
    if minxvalue is None:
        Xmin = min(X)
    else:
        Xmin = minxvalue
    if maxxvalue is None:
        Xmax = max(X)
    else:
        Xmax = maxxvalue
    X = aspect_ratio[0]*(X-Xmin)/(Xmax-Xmin)

    Y = df[ycol].to_numpy()
    if minyvalue is None:
        Ymin = min(Y)
    else:
        Ymin = minyvalue
    if maxyvalue is None:
        Ymax = max(Y)
    else:
        Ymax = maxyvalue
    Y = aspect_ratio[1]*(Y-Ymin)/(Ymax-Ymin)

    Z = df[zcol].to_numpy()
    if minzvalue is None:
        Zmin = min(Z)
    else:
        Zmin = minzvalue
    if maxzvalue is None:
        Zmax = max(Z)
    else:
        Zmax = maxzvalue
    Z = aspect_ratio[2]*(Z-Zmin)/(Zmax-Zmin)

    # mesh vertices and faces
    verts = list(zip(X, Y, Z))

    xsize = df[xcol].nunique()
    ysize = df[ycol].nunique()
    faces = [(xsize*j+i, xsize*j+i+1, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)] + \
            [(xsize*j+i, xsize*j+xsize+i, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)]

    mesh = Mesh([verts, faces])
    mesh.pointdata['meter'] = Z     # you must first associate numerical data to mesh points
    mesh.pointdata.select('meter')  # and then make them "active"
    mesh.cmap(palette)

    isol = mesh.isolines(n=num_isolines).color('w')
    isol.lw(3)

    # set up camera
    focal_point = (0.5*aspect_ratio[0], 0.5*aspect_ratio[1], 0.5*aspect_ratio[2])
    half_diagonal = 0.5*math.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2 + aspect_ratio[2]**2)

    if type(camera) is tuple:
        # azimuth and altitude are given as camera[0] and camera[1]
        cam_pos = (0.5*aspect_ratio[0] + distance_multiplier * half_diagonal *
                   math.cos((camera[0]-90)*math.pi/180) * math.cos(camera[1]*math.pi/180),
                   0.5*aspect_ratio[1] + distance_multiplier * half_diagonal *
                   math.sin((camera[0]-90)*math.pi/180) * math.cos(camera[1]*math.pi/180),
                   0.5*aspect_ratio[2] + distance_multiplier * half_diagonal *
                   math.sin(camera[1]*math.pi/180))
        cam = dict(
            position=cam_pos,
            focal_point=focal_point,
            viewup=(0, 0, 1),
            distance=distance_multiplier * half_diagonal,
            # clipping_range=(1.0, 6.0),
        )
    elif type(camera) is dict:
        # vedo camera dictionary is provided
        cam = camera
    else:
        raise ValueError('Camera parameter should be either tuple or dict')

    # set up light
    if light is None:
        light_pos = (0.5*aspect_ratio[0], 0.5*aspect_ratio[1], 1.5*aspect_ratio[2])
        light = Light(pos=light_pos, focal_point=focal_point, c='w', intensity=1)

    # set up positions of coordinate planes by comparing cam['position'] and cam['focal_point']
    if cam['position'][0] < cam['focal_point'][0]:
        # looking from the left
        yzgrid = False
        yzgrid2 = True
        zshift_along_x = 1
        yshift_along_x = 0
        ylabel_offset = 0.8
        ylabel_justify = 'right-center'
        ytitle_offset = 0.0275
        zlabel_offset = -0.8
        zlabel_justify = 'center-left'
        ztitle_offset = -0.285
    else:
        # looking from the right
        yzgrid = True
        yzgrid2 = False
        zshift_along_x = 0
        yshift_along_x = 1
        ylabel_offset = -0.8
        ylabel_justify = 'center-left'
        ytitle_offset = -0.25
        zlabel_offset = 0.6
        zlabel_justify = 'right-center'
        ztitle_offset = -0.01

    if cam['position'][1] < cam['focal_point'][2]:
        # looking from front
        zxgrid = False
        zxgrid2 = True
        xshift_along_y = 0
        xlabel_offset = 0.8
        xtitle_offset = 0.15
        zshift_along_y = 0
    else:
        # looking from behind
        zxgrid = True
        zxgrid2 = False
        xshift_along_y = 1
        xlabel_offset = -1.5
        xtitle_offset = -0.45
        zshift_along_y = 1

    if cam['position'][2] < cam['focal_point'][2]:
        # looking from below
        xygrid = False
        xygrid2 = True
        xshift_along_z = 1
        yshift_along_z = 1
    else:
        # looking from above
        xygrid = True
        xygrid2 = False
        xshift_along_z = 0
        yshift_along_z = 0

    plt = Plotter(N=1,
                  size=diagram_size,
                  axes=dict(xtitle=xcol,
                            xtitle_offset=xtitle_offset,
                            xygrid=xygrid,
                            xygrid2=xygrid2,
                            xshift_along_y=xshift_along_y,
                            xshift_along_z=xshift_along_z,
                            xlabel_offset=xlabel_offset,
                            ytitle=ycol,
                            ytitle_offset=ytitle_offset,
                            ylabel_offset=ylabel_offset,
                            ylabel_justify=ylabel_justify,
                            yzgrid=yzgrid,
                            yzgrid2=yzgrid2,
                            yshift_along_x=yshift_along_x,
                            yshift_along_z=yshift_along_z,
                            zshift_along_x=zshift_along_x,
                            zshift_along_y=zshift_along_y,
                            ztitle=zcol,
                            ztitle_offset=ztitle_offset,
                            zlabel_offset=zlabel_offset,
                            zlabel_justify=zlabel_justify,
                            zxgrid=zxgrid,
                            zxgrid2=zxgrid2,
                            xrange=(-padding, aspect_ratio[0]+padding),
                            yrange=(-padding, aspect_ratio[1]+padding),
                            zrange=(-padding, aspect_ratio[2]+padding),
                            x_values_and_labels=[(i, f'{Xmin + i*(Xmax-Xmin)/aspect_ratio[0]:.2f}')
                                                 for i in np.linspace(0, aspect_ratio[0], numxticks)],
                            y_values_and_labels=[(i, f'{Ymin + i*(Ymax-Ymin)/aspect_ratio[1]:.2f}')
                                                 for i in np.linspace(0, aspect_ratio[1], numyticks)],
                            z_values_and_labels=[(i, f'{Zmin + i*(Zmax-Zmin)/aspect_ratio[2]:.2f}')
                                                 for i in np.linspace(0, aspect_ratio[2], numzticks)],
                            axes_linewidth=3,
                            grid_linewidth=2,
                            number_of_divisions=max((numxticks, numyticks, numzticks)),
                            text_scale=1.5,
                            **kwargs)).parallel_projection(value=True)

    plt.show(mesh, isol, light, camera=cam, interactive=interactive, zoom=1)

    if filename is None:
        filename = "diagram_" + xcol + "_" + ycol + "_" + zcol
    # make sure filename is valid...
    filename = str(filename)
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\s-]', '', filename.lower())
    filename = re.sub(r'[-\s]+', '-', filename).strip('-_')
    filename = filename + ".png"

    plt.screenshot(filename)
    plt.close()


if __name__=="__main__":
    # print(MIPT(n=10, dim=2))
    # print(MIPT_extend(np.array([[0.5, 0.5]]), n=9))
    # print(hypercube_to_indices(np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]), np.array([2,3,4])))
    # print(indices_to_hypercube(np.array([[0],[1],[2]]), [4])
    # df = sample_and_simulate_overhangs('m1Dubai.idf','m1Dubai.epw',
    #                                    'Office_Cell_Wall_South_Window',
    #                                    [0.25, 0.5, 0.75, 1],
    #                                    [0.25, 0.5, 0.75, 1],
    #                                    5.0,
    #                                    8)
    # # height and depth are the first two columns,
    # # the remaining ones are the meters for which the XGBoost ensembles are trained
    # cols = df.columns.to_list()
    # models = train_model_ensembles(df, cols[0:2], cols[2:])
    # df2 = predict_meters(models, {'height': np.linspace(0.25, 1, 50).round(decimals=2),
    #                               'depth': np.linspace(0.25, 1, 50).round(decimals=2)})
    # df = pd.read_csv('starting_case_predictions.csv')
    # make_3d_diagram(df, 'depth', 'height', 'primary [kWh/m2]',
    #                 aspect_ratio=[1.6, 0.5, 1],
    #                 numxticks=9,
    #                 minyvalue=0.0, maxyvalue=0.5, numyticks=6,
    #                 minzvalue=250, maxzvalue=300, numzticks=6)
