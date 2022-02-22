import librosa
import numpy as np
import scipy.ndimage

def scluster(embeddings):
    '''
    reference: 
    https://librosa.org/doc/0.9.1/auto_examples/plot_segmentation.html?highlight=feature%20sync
    Brian McFee, Daniel P.W. Ellis. “Analyzing song structure with spectral clustering”, 
    15th International Society for Music Information Retrieval Conference, 2014.

    Both S_rep and S_loc use embeddings as the input feature
    '''
    # S_rep, calculated from beat-sychronous CQT
    R = librosa.segment.recurrence_matrix(embeddings, width=3, mode='affinity', sym=True)

    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # S_loc, from MFCC13
    path_distance = np.sum(np.diff(embeddings, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path

    # compute normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)


    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # If we want k clusters, use the first k normalized eigenvectors.
    # Fun exercise: see how the segmentation changes as you vary k
    k = 5
    X = evecs[:, :k] / Cnorm[:, k-1:k]

    return X