# plotting related settings

from matplotlib import rc
from matplotlib.ticker import (
    MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib import rcParams

rcParams['figure.dpi'] = 100
rcParams['figure.figsize'] = [6., 4.]
rcParams['axes.linewidth'] = 1.
rcParams['font.size'] = 15.

axisFace = '#323A48'
figureFace = '#323A48'
textColor = '#DBE1EA'
edgeColor = '#92A2BD'
gridColor = '#3F495A'
notebook_bg = '#1A2028'
yellow = '#FFEC8E'
orange = '#ff7f0e'
red = '#e17e85'
magenta = '#e07a7a'
violet = '#be86e3'
blue = '#1f77b4'
cyan = '#4cb2ff'
green = '#61ba86'

rcParams['lines.linewidth'] = 1.25


# Latex related settings
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{amsmath}   \usepackage{mathrsfs} \usepackage{amssymb}')
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'serif'

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
