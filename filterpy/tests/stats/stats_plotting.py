import numpy as np
import matplotlib.pyplot as plt
import math
from math import cos, sin
import warnings
from scipy.stats import norm
from filterpy.stats import covariance_ellipse

def plot_discrete_cdf(xs, ys, ax=None, xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    xs : list-like of scalars
        x values corresponding to the values in `y`s. Can be `None`, in which
        case range(len(ys)) will be used.

    ys : list-like of scalars
        list of probabilities to be plotted which should sum to 1.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """

    if ax is None:
        ax = plt.gca()

    if xs is None:
        xs = range(len(ys))
    ys = np.cumsum(ys)
    ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_gaussian_cdf(mean=0., variance=1.,
                      ax=None,
                      xlim=None, ylim=(0., 1.),
                      xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 0.
        variance for the normal distribution.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """
    if ax is None:
        ax = plt.gca()

    sigma = math.sqrt(variance)
    n = norm(mean, sigma)
    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    cdf = n.cdf(xs)
    ax.plot(xs, cdf, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_gaussian_pdf(mean=0.,
                      variance=1.,
                      std=None,
                      ax=None,
                      mean_line=False,
                      xlim=None, ylim=None,
                      xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution PDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the probability density.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 1., optional
        variance for the normal distribution.

    std: scalar, default=None, optional
        standard deviation of the normal distribution. Use instead of
        `variance` if desired

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    mean_line : boolean
        draws a line at x=mean

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if variance is not None and std is not None:
        raise ValueError('Specify only one of variance and std')

    if variance is None and std is None:
        raise ValueError('Specify variance or std')

    if variance is not None:
        std = math.sqrt(variance)

    n = norm(mean, std)

    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    ax.plot(xs, n.pdf(xs), label=label)
    ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if mean_line:
        plt.axvline(mean)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_gaussian(mean=0., variance=1.,
                  ax=None,
                  mean_line=False,
                  xlim=None,
                  ylim=None,
                  xlabel=None,
                  ylabel=None,
                  label=None):
    """
    DEPRECATED. Use plot_gaussian_pdf() instead. This is poorly named, as
    there are multiple ways to plot a Gaussian.
    """

    warnings.warn('This function is deprecated. It is poorly named. '\
                  'A Gaussian can be plotted as a PDF or CDF. This '\
                  'plots a PDF. Use plot_gaussian_pdf() instead,',
                  DeprecationWarning)
    return plot_gaussian_pdf(mean, variance, ax, mean_line, xlim, ylim, xlabel,
                             ylabel, label)

def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.

    Parameters
    ----------
    cov : ndarray
        covariance matrix

    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)

    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest

    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]

def plot_3d_covariance(mean, cov, std=1.,
                       ax=None, title=None,
                       color=None, alpha=1.,
                       label_xyz=True,
                       N=60,
                       shade=True,
                       limit_xyz=True,
                       **kwargs):
    """
    Plots a covariance matrix `cov` as a 3D ellipsoid centered around
    the `mean`.

    Parameters
    ----------

    mean : 3-vector
        mean in x, y, z. Can be any type convertable to a row vector.

    cov : ndarray 3x3
        covariance matrix

    std : double, default=1
        standard deviation of ellipsoid

    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3d axis will be generated
        for the current figure

    title : str, optional
        If provided, specifies the title for the plot

    color : any value convertible to a color
        if specified, color of the ellipsoid.

    alpha : float, default 1.
        Alpha value of the ellipsoid. <1 makes is semi-transparent.

    label_xyz: bool, default True

        Gives labels 'X', 'Y', and 'Z' to the axis.

    N : int, default=60
        Number of segments to compute ellipsoid in u,v space. Large numbers
        can take a very long time to plot. Default looks nice.

    shade : bool, default=True
        Use shading to draw the ellipse

    limit_xyz : bool, default=True
        Limit the axis range to fit the ellipse

    **kwargs : optional
        keyword arguments to supply to the call to plot_surface()
    """

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # force mean to be a 1d vector no matter its shape when passed in
    mean = np.atleast_2d(mean)
    if mean.shape[1] == 1:
        mean = mean.T

    if not(mean.shape[0] == 1 and mean.shape[1] == 3):
        raise ValueError('mean must be convertible to a 1x3 row vector')
    mean = mean[0]

    # force covariance to be 3x3 np.array
    cov = np.asarray(cov)
    if cov.shape[0] != 3 or cov.shape[1] != 3:
        raise ValueError("covariance must be 3x3")

    # The idea is simple - find the 3 axis of the covariance matrix
    # by finding the eigenvalues and vectors. The eigenvalues are the
    # radii (squared, since covariance has squared terms), and the
    # eigenvectors give the rotation. So we make an ellipse with the
    # given radii and then rotate it to the proper orientation.

    eigval, eigvec = _eigsorted(cov, asc=True)
    radii = std * np.sqrt(np.real(eigval))

    if eigval[0] < 0:
        raise ValueError("covariance matrix must be positive definite")


    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v)) * radii[0]
    y = np.outer(np.sin(u), np.sin(v)) * radii[1]
    z = np.outer(np.ones_like(u), np.cos(v)) * radii[2]

    # rotate data with eigenvector and center on mu
    a = np.kron(eigvec[:, 0], x)
    b = np.kron(eigvec[:, 1], y)
    c = np.kron(eigvec[:, 2], z)

    data = a + b + c
    N = data.shape[0]
    x = data[:,   0:N]   + mean[0]
    y = data[:,   N:N*2] + mean[1]
    z = data[:, N*2:]    + mean[2]

    fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,
                    rstride=3, cstride=3, linewidth=0.1, alpha=alpha,
                    shade=shade, color=color, **kwargs)

    # now make it pretty!

    if label_xyz:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if limit_xyz:
        r = radii.max()
        ax.set_xlim(-r + mean[0], r + mean[0])
        ax.set_ylim(-r + mean[1], r + mean[1])
        ax.set_zlim(-r + mean[2], r + mean[2])

    if title is not None:
        plt.title(title)

    #pylint: disable=pointless-statement
    Axes3D #kill pylint warning about unused import

    return ax


def plot_covariance_ellipse(
        mean, cov=None, variance=1.0, std=None,
        ellipse=None, title=None, axis_equal=True, show_semiaxis=False,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):
    """
    Deprecated function to plot a covariance ellipse. Use plot_covariance
    instead.

    See Also
    --------

    plot_covariance
    """

    warnings.warn("deprecated, use plot_covariance instead", DeprecationWarning)
    plot_covariance(mean=mean, cov=cov, variance=variance, std=std,
                    ellipse=ellipse, title=title, axis_equal=axis_equal,
                    show_semiaxis=show_semiaxis, facecolor=facecolor,
                    edgecolor=edgecolor, fc=fc, ec=ec, alpha=alpha,
                    xlim=xlim, ylim=ylim, ls=ls)
    

def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.

    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)

    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std


    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)



def plot_covariance(
        mean, cov=None, variance=1.0, std=None, interval=None,
        ellipse=None, title=None, axis_equal=True,
        show_semiaxis=False, show_center=True,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):
    """
    Plots the covariance ellipse for the 2D normal defined by (mean, cov)

    `variance` is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses. Alternatively, use std for the
    standard deviation, in which case `variance` will be ignored.

    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.

    You may provide either cov or ellipse, but not both.

    Parameters
    ----------

    mean : row vector like (2x1)
        The mean of the normal

    cov : ndarray-like
        2x2 covariance matrix

    variance : float, default 1, or iterable float, optional
        Variance of the plotted ellipse. May specify std or interval instead.
        If iterable, such as (1, 2**2, 3**2), then ellipses will be drawn
        for all in the list.


    std : float, or iterable float, optional
        Standard deviation of the plotted ellipse. If specified, variance
        is ignored, and interval must be `None`.

        If iterable, such as (1, 2, 3), then ellipses will be drawn
        for all in the list.

    interval : float range [0,1), or iterable float, optional
        Confidence interval for the plotted ellipse. For example, .68 (for
        68%) gives roughly 1 standand deviation. If specified, variance
        is ignored and `std` must be `None`

        If iterable, such as (.68, .95), then ellipses will be drawn
        for all in the list.


    ellipse: (float, float, float)
        Instead of a covariance, plots an ellipse described by (angle, width,
        height), where angle is in radians, and the width and height are the
        minor and major sub-axis radii. `cov` must be `None`.

    title: str, optional
        title for the plot

    axis_equal: bool, default=True
        Use the same scale for the x-axis and y-axis to ensure the aspect
        ratio is correct.

    show_semiaxis: bool, default=False
        Draw the semiaxis of the ellipse

    show_center: bool, default=True
        Mark the center of the ellipse with a cross

    facecolor, fc: color, default=None
        If specified, fills the ellipse with the specified color. `fc` is an
        allowed abbreviation

    edgecolor, ec: color, default=None
        If specified, overrides the default color sequence for the edge color
        of the ellipse. `ec` is an allowed abbreviation

    alpha: float range [0,1], default=1.
        alpha value for the ellipse

    xlim: float or (float,float), default=None
       specifies the limits for the x-axis

    ylim: float or (float,float), default=None
       specifies the limits for the y-axis

    ls: str, default='solid':
        line style for the edge of the ellipse
    """

    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt

    if cov is not None and ellipse is not None:
        raise ValueError('You cannot specify both cov and ellipse')

    if cov is None and ellipse is None:
        raise ValueError('Specify one of cov or ellipse')

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title(title)

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls)
        ax.add_patch(e)
    x, y = mean
    if show_center:
        plt.scatter(x, y, marker='+', color=edgecolor)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height/4, width/4
        plt.plot([x, x+ h*cos(a+np.pi/2)], [y, y + h*sin(a+np.pi/2)])
        plt.plot([x, x+ w*cos(a)], [y, y + w*sin(a)])
