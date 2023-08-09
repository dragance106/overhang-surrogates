import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from itertools import combinations
import timeit

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


def mipt(n, dim=2, alpha=0.5, k=100):
    """
    Implementation of the Crombecq's mc-intersite-proj-th sampling scheme,
    in which the new candidate points are generated over the whole design space [0,1]^dim.

    :param n:     the number of sample points to be generated
    :param dim:   the dimension of the design space
    :param alpha: the tolerance parameter for the minimum projected distance:
                  any candidate points with projected distance smaller than alpha/n is discarded
    :param k:     the number of candidate points to be generated in the i-th iteration
                  (after i-1 points have already been generated)
                  will be equal to k*i

    :return:      the sequence of n sample points from [0,1]^dim.
    """

    # placeholder for the sampled points
    sample = np.zeros((n, dim))

    # the first point is just randomly generated
    rng = np.random.default_rng()
    sample[0] = rng.random((dim,))

    s=1
    while s<n:
    # for s in range(1, n):
        # generate k*s random candidate points
        candidates = rng.random((k*s, dim))

        # from each candidate point to the s already selected sample points
        # compute projected distance, intersite distance and score them
        dmin = alpha/(s+1)
        zeros = np.zeros((k*s,))
        prd = proj_dist(candidates, sample[:s])
        ind = inter_dist(candidates, sample[:s])
        scores = np.where(prd<dmin, zeros, ind)

        # note that there is a nonzero possibility here that
        # no candidate was sufficiently far from the already selected sample points
        if np.max(scores) < 1e-9:
            continue

        # here we solely depend on alpha=0.5 to do its miracle
        # (alpha=0 means that we do not take projected_distance into account,
        #  while alpha=1 means that we will reject almost all candidates,
        #  and end up with the same result as with alpha=0...)

        # select the best candidate point and add it to the sample
        sample[s] = candidates[np.argmax(scores)]
        s += 1

    # n points have been now sampled
    return sample


def mipt_full(n, dim=2, alpha=0.5, k=100, negligible=1e-6):
    """
    Implementation of the Crombecq's mc-intersite-proj-th sampling scheme,
    in which the new candidate points are generated only within the allowed intervals,
    obtained after subtracting from [0,1]^dim
    the hypercubes covering the minimum projected distance around the already selected sample points.

    :param n:     the number of sample points to be generated
    :param dim:   the dimension of the design space
    :param alpha: the tolerance parameter for the minimum projected distance:
                  any candidate points with projected distance smaller than alpha/n is discarded
    :param k:     the number of candidate points to be generated in the i-th iteration
                  (after i-1 points have already been generated)
                  will be equal to k*i
    :param negligible:   the value considered negligible when mutually comparing
                         boundaries of different intervals

    :return:      the sequence of n sample points from [0,1]^dim.
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


from skopt.sampler import Lhs
lhs = Lhs(criterion='maximin', iterations=10000)

def LHS_maximin(n, dim=2):
    """
    Bridge between a general sampling call in train_xgb.py and
    skopt.sampler.Lhs from the scikit-optimize package.
    """

    # lhs = Lhs(criterion="maximin", iterations=1000)
    return lhs.generate([[0.0,1.0]]*dim, n)


if __name__=="__main__":
    n = 100

    tic = timeit.default_timer()
    sample1 = mipt(n)
    min_inter1 = inter_quality(sample1)
    min_proj1 = proj_quality(sample1)
    toc = timeit.default_timer()
    print(f'MIPT time: {toc-tic} sec.')

    tic = toc
    sample2 = mipt_full(n)
    min_inter2 = inter_quality(sample2)
    min_proj2 = proj_quality(sample2)
    toc = timeit.default_timer()
    print(f'MIPT-full time: {toc-tic} sec.')

    tic = toc
    sample3 = np.array(LHS_maximin(n))
    min_inter3 = inter_quality(sample3)
    min_proj3 = proj_quality(sample3)
    toc = timeit.default_timer()
    print(f'LHS-maximin time: {toc-tic} sec.')

    # extend samples with the origin indicator
    ones = np.ones((n,1))
    twos = 2*np.ones((n,1))
    threes = 3*np.ones((n,1))
    all_points = np.block([[sample1, ones], [sample2, twos], [sample3, threes]])

    plt.figure(1, figsize=(6, 6), dpi=300, layout='constrained')
    plt.title(label=f'{n} samples, inter1={min_inter1:.5f}, proj1={min_proj1:.5f}, inter2={min_inter2:.5f}, proj2={min_proj2:.5f}, inter3={min_inter3:.5f}, proj3={min_proj3:.5f}', fontsize=7, color='darkred')
    sb.scatterplot(x=all_points[:,0], y=all_points[:,1], hue=all_points[:,2])
    plt.savefig(f'sample {n}.png')
    plt.cla()
