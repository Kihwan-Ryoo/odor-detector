import numpy as np
import pandas as pd
import cv2
from random import *
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def colour_to_thresh(frame, block_size = 31, offset = 25):
    """
    This function retrieves a video frame and preprocesses it for object tracking.
    The code blurs image to reduce noise, converts it to greyscale and then returns a 
    thresholded version of the original image.
    
    Parameters
    ----------
    frame: ndarray, shape(n_rows, n_cols, 3)
        source image containing all three colour channels
    block_size: int(optional), default = 31
        block_size determines the width of the kernel used for adaptive thresholding.
        Note: block_size must be odd. If even integer is used, the programme will add
        1 to the block_size to make it odd.
    offset: int(optional), default = 25
        constant subtracted from the mean value within the block
        
    Returns
    -------
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    """
    blur = cv2.blur(frame, (5,5))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, offset)
    return thresh

def detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area = 0, max_area = 10000):
    """
    This function detects contours, thresholds them based on area and draws them.
    
    Parameters
    ----------
    frame: ndarray, shape(n_rows, n_cols, 3)
        source image containing all three colour channels
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
    min_area: int
        minimum area threhold used to detect the object of interest
    max_area: int
        maximum area threhold used to detect the object of interest
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    # Detect contours and draw them based on specified area thresholds
    if int(cv2.__version__[0]) == 3:
        img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = list(contours) # Kihwan 2022/01/19 (TypeError: 'tuple' object doesn't support item deletion)

    final = frame.copy()

    i = 0
    meas_last = meas_now.copy()
    del meas_now[:]
    while i < len(contours):
        area = cv2.contourArea(contours[i])
        if area < min_area or area > max_area:
            del contours[i]
        else:
            M = cv2.moments(contours[i])
            if M['m00'] != 0:
                # formula for center of gravity
            	cx = M['m10']/M['m00']
            	cy = M['m01']/M['m00']
            else:
            	cx = 0
            	cy = 0
            meas_now.append([cx, cy])
            i += 1
    return final, contours, meas_last, meas_now

def apply_k_means(contours, n_inds, meas_now):
    """
    This function applies the k-means clustering algorithm to separate merged
    contours. The algorithm is applied when detected contours are fewer than
    expected objects(number of animals) in the scene.
    
    Parameters
    ----------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    n_inds: int
        total number of individuals being tracked
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    del meas_now[:]
    # Clustering contours to separate individuals
    myarray = np.vstack(contours)
    myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])

    kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)
    l = len(kmeans.cluster_centers_)

    for i in range(l):
        x = int(tuple(kmeans.cluster_centers_[i])[0])
        y = int(tuple(kmeans.cluster_centers_[i])[1])
        meas_now.append([x,y])
    return contours, meas_now

def hungarian_algorithm(meas_last, meas_now):
    """
    The hungarian algorithm is a combinatorial optimisation algorithm used
    to solve assignment problems. Here, we use the algorithm to reduce noise
    due to ripples and to maintain individual identity. This is accomplished
    by minimising a cost function; in this case, euclidean distances between 
    points measured in previous and current step. The algorithm here is written
    to be flexible as the number of contours detected between successive frames
    changes. However, an error will be returned if zero contours are detected.
   
    Parameters
    ----------
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    row_ind: array, dtype=int64
        individual identites arranged according to input ``meas_last``
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    """
    meas_last = np.array(meas_last)
    meas_now = np.array(meas_now)
    if meas_now.shape != meas_last.shape:
        if meas_now.shape[0] < meas_last.shape[0]:
            while meas_now.shape[0] != meas_last.shape[0]:
                meas_last = np.delete(meas_last, meas_last.shape[0]-1, 0)
        else:
            result = np.zeros(meas_now.shape)
            result[:meas_last.shape[0], :meas_last.shape[1]] = meas_last
                # When zero contours are detected,
                # IndexError: tuple index out of range
            meas_last = result

    meas_last = list(meas_last)
    meas_now = list(meas_now)
    if(len(meas_last)>0):
        cost = cdist(meas_last, meas_now) # Compute distance between each pair of the two collections of inputs.
        '''
        cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)
        '''
        cost[cost > 20]=10000
        row_ind, col_ind = linear_sum_assignment(cost) # Solve the linear sum assignment problem
        '''
        linear_sum_assignment(cost_matrix)
            Parameters - cost_matrix: array
                            The cost matrix of the bipartite graph
            Returns - row_ind, col_ind: array
                        An array of row indices and one of corresponding column indices giving the optimal assignment.
                        The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum()
                        The row indices will be sorted; in the case of a square cost matrix they will be equal to np.arange(cost_matrix.shape[0])
        The linear sum assignment problem is also known as minimum weight matching in bipartite graphs.
        A problem instance is described by a matrix C,
        where each Cij is the cost of matching vertex i of the first partite set (a "worker") and vertex j of the second set (a "job").
        The goal is to find a complete assignment of workers to jobs of minimal cost.
        The method used is the Hungarian algorithm, also known as the Munkres or Kuhn-Munkres algorithm.
        https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        '''
    else:
        row_ind=[]
        col_ind=[]
        
    return row_ind, col_ind

def reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot):
    """
    This function reorders the measurements in the current frame to match
    identity from previous frame. This is done by using the results of the
    hungarian algorithm from the array col_inds.
    
    Parameters
    ----------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    colours: list, tuple
        list of tuples that represent colours used to assign individual identities
    n_inds: int
        total number of individuals being tracked
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.core.frame.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    mot: bool
        this boolean determines if we apply the alogrithm to a multi-object
        tracking problem
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    """
    t_id=list(range(1,501)) #101
    t_id=list(map(str,t_id))


    # Reorder contours based on results of the hungarian algorithm
    equal = np.array_equal(col_ind, list(range(len(col_ind)))) # True if two arrays have the same shape and elements, False otherwise.
    if equal == False:
        current_ids = col_ind.copy()
        reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])] # 오름차순
        meas_now = [x for (y, x) in sorted(zip(reordered, meas_now))]

    # Draw centroids
    if mot == False:
        for i in range(len(meas_now)):
            if colours[i%4] == (0,0,255):
                cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 2, colours[i%4], -1, cv2.LINE_AA)
    else:
        for i in range(len(meas_now)):
            cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 2, colours[i%len(meas_now)], -1, cv2.LINE_AA)
        
    return final, meas_now, df


def reject_outliers(data, m):
    """
    This function removes any outliers from presented data.
    
    Parameters
    ----------
    data: pandas.Series
        a column from a pandas dataframe that needs smoothing
    m: float
        standard deviation cutoff beyond which, datapoint is considered as an outlier
        
    Returns
    -------
    index: ndarray
        an array of indices of points that are not outliers
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)