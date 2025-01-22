import numpy as np

def cross_corr(sig1, sig2, dt, norm=True, no_bias=True):
    """Homemade cross-correlation function.

    Computes the cross-correlation time-series of two signals. Function was
    designed for signals of same length. If signals of different sizes are
    used, correlation lag might be strange, use at your own risk.

    Args:
        sig1, sig2 (1D arrays): Signals to cross-correlate, dimensions N and M. Use same discretization lenght `dt` for best results.
        dt (float): Time step used in both signals.
        norm (bool, optional): By default, normalizes the signals before cross-correlating them. `norm = False` turns it off.
        no_bias (bool, optional): By default, removes the bias due to the variable number of points used to compute the correlation at each lag. This option also removes one fifth of the cross-correlation at each end, to get rid of edge effects due to unbiasing.

    Returns:
        corr (1D array): Cross-correlation of the input signals, dimension N + M - 1.
        lag (1D array): Time lag for each value of the cross-correlation, dimension N + M - 1. Centering around 0 is only ensured if sig1 and sig2 are the same size.
    """

    # >> Normalize signals
    if norm:
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1)*len(sig1))
        sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)

    # >> Compute correlation
    corr = np.correlate(sig1.astype(float), sig2.astype(float), 'full')

    # >> Normalize
    if norm:
        corr = corr.astype(float) / (np.linalg.norm(sig1)*np.linalg.norm(sig2))

    # >> Remove bias due to number of points varying at each lag
    if no_bias:
        bias = cross_corr_bias(sig1, sig2)
        corr = corr/bias

    # >> Compute time lag of autocorrelation
    lag = np.arange(0, len(sig1) + len(sig2) - 1) - (len(sig1) - 1)
    lag = lag.astype(float) * dt

    # >> Removes both ends of lag and correlation, to get rid of edge effects
    # due to removing the bias
    if no_bias:
        valid = (abs(lag) < 4/5*max(lag))
        lag = lag[valid]
        corr = corr[valid]

    return corr, lag


#------------------------------------------------------------------------------

def cross_corr_bias(sig1, sig2):
    """Computes the bias of at each lag of the cross correlation.

    Args:
        sig1, sig2 (1D array): Signals to correlate, dimension N and M.

    Returns:
        bias (1D array): The bias vector, corresponding to all values of lag (both negative and positive), dimension N + M - 1.

    """
    N = len(sig1)
    M = len(sig2)

    # >> Bias for part where signals overlap entirely
    bias_full = [1 for ii in range(max(M, N) - min(M, N) + 1)]

    # >> Bias for part before full overlap
    bias_bef = [ii/min(N, M) for ii in range(1, min(M, N))]

    # >> Bias for part after full overlap
    bias_aft = [ii/min(N, M) for ii in range(min(M, N) - 1, 0, -1)]

    # >> Concatenate
    bias = np.concatenate((bias_bef, bias_full, bias_aft))

    return bias


def correlation_matrix(trm, parameters, verbose=False):
    """Compute the cross correlation matrix

    The cross-correlation matrix computes the cross-correlation coefficient (allowing for a lag) and lag between activity time series of tremor along strike (binned).

    Args:
        trm (pandas dataframe): Tremor dataframe. Should contain a location column `xf` and a `day` column.
        parameters (dict): Parameters of the computation. See code for details.

    Returns:
        cc_matrix, lag_matrix: 2D arrays with the cross-correlation value and the lag value.
    """
    # Get parameters
    dx_bin = parameters['dx_bin']
    xmin_bin = parameters['xmin_bin']
    xmax_bin = parameters['xmax_bin']
    delta = parameters['delta']
    norm = parameters['norm']
    no_bias = parameters['no_bias']
    look_both_ways = parameters['look_both_ways']
    v_min = parameters['v_min']

    x_bin_edges = np.arange(xmin_bin, xmax_bin + dx_bin, dx_bin)

    # Make the activity count matrix: each row corresponds to a spatial bin along-strike, each column to a temporal bin
    count_time = np.arange(0, trm.day.max()+delta, delta)
    counts = np.zeros((len(x_bin_edges)-1, len(count_time)-1))

    for ii in range(len(x_bin_edges)-1):
        x_min = x_bin_edges[ii]
        x_max = x_bin_edges[ii+1]
        trm_in_bin = trm.xf.between(x_min, x_max)
        counts[ii, :] = np.histogram(trm[trm_in_bin].day.values, bins=count_time)[0]

    # Compute the cross-correlation matrix
    cc_matrix = np.zeros((len(x_bin_edges)-1, len(x_bin_edges)-1))
    lag_matrix = np.zeros((len(x_bin_edges)-1, len(x_bin_edges)-1))

    for ii in range(len(x_bin_edges)-1):
        if verbose: print('{:d}/{:d}'.format(ii, len(x_bin_edges)-1), end='\r')
        for jj in range(len(x_bin_edges)-1):
            if jj <= ii:  # as the matrix is symetrical, only compute for one pair of indices
                if ~np.any(counts[ii, :]) or ~np.any(counts[jj, :]):  # if no activity in one of the bins, set to nan
                    cc_matrix[ii, jj] = cc_matrix[jj, ii] = np.nan
                    lag_matrix[ii, jj] = lag_matrix[jj, ii] = np.nan
                else:
                    # Compute cross correlation
                    cc, lag = cross_corr(counts[ii, :], counts[jj, :], delta, norm=norm, no_bias=no_bias)
                    
                    if look_both_ways:  # averages the correlation at positive and negative lag
                        cc = (cc + cc[::-1]) / 2
                        cc = cc[lag >= 0]
                        lag = lag[lag >= 0]
                    
                    # Look for maximum in a time window around lag=0, and store the results in the matrices
                    in_window = np.abs(lag) <= abs((x_bin_edges[ii+1] + x_bin_edges[ii])/2 - (x_bin_edges[jj+1] + x_bin_edges[jj])/2) / v_min
                    cc_matrix[ii, jj] = cc_matrix[jj, ii] = np.max(cc[in_window])
                    lag_matrix[ii, jj] = lag_matrix[jj, ii] = lag[in_window][np.argmax(cc[in_window])]
    
    return cc_matrix, lag_matrix

def correlation_distance(cc_matrix, cc_thr, x_bin_edges):
    """Compute well-correlated width from cross-correlation matrix

    Args:
        cc_matrix (ndarray, 2D): Cross correlation matrix (Nx, Nx).
        cc_thr (float): Threshold correlation defining "well-correlated".
        x_bin_edges (ndarray, 1D): Bin edges used to compute the matrix (Nx+1).

    Returns:
        corr_length, left_width, right_width: Arrays containing the total correlation length, and its components in each direction along x.
    """
    left_width = []
    right_width = []

    # Run through rows of the matrix, and look how far from diagonal terms exceed cc_thr
    for ii in range(len(x_bin_edges)-1):
        # Look right
        jj = ii
        while (jj < len(x_bin_edges)-1) and (cc_matrix[ii, jj] > cc_thr):
            jj += 1
        right_width.append(x_bin_edges[jj-1] - x_bin_edges[ii])

        # Look left
        jj = ii
        while (jj >= 0) and (cc_matrix[ii, jj] > cc_thr):
            jj -= 1
        left_width.append(x_bin_edges[ii] - x_bin_edges[jj+1])

    left_width = np.array(left_width).astype(float)
    right_width = np.array(right_width).astype(float)

    # The correlation length is the total well-correlated width around the diagonal
    corr_length = np.array(left_width) + np.array(right_width)

    return corr_length, left_width, right_width