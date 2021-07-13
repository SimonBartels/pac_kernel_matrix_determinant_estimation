import os
import numpy as np

TIKZ_OUTPUT_PATH = os.path.join(os.getcwd(), "tikz")

# constants
lowest_r = 1e-4
diagonal_precision_label = "diagonal precision"
diagonal_precision_color = 'black'
diagonal_precision_style = '--'

# colors (credits to Philipp Hennig)
mpg = np.array([0,0.4717,0.4604])
dre = np.array([0.4906,0,0])
ora = np.array([255,153,51]) / 255
blu = np.array([0,0,0.509])
gra = 0.5 * np.ones(3)

lightmpg = [1,1,1] - 0.5 * ([1,1,1] - mpg)
lightdre = [1,1,1] - 0.5 * ([1,1,1] - dre)
lightblu = [1,1,1] - 0.5 * ([1,1,1] - blu)
lightora = [1,1,1] - 0.5 * ([1,1,1] - ora)

cya = lightmpg

mpg2white = np.ones([2024, 3]) - np.sqrt(np.linspace(0,0.6,2024)[:, np.newaxis]).dot((np.ones([3, 1])-mpg[:, np.newaxis]).T)
dre2white = np.ones([2024, 3]) - np.sqrt(np.linspace(0,0.6,2024)[:, np.newaxis]).dot((np.ones([3, 1])-dre[:, np.newaxis]).T)
cmapMPGDre_ = np.vstack([np.flipud(mpg2white), dre2white])
from matplotlib.colors import LinearSegmentedColormap
cmapMPGDre = LinearSegmentedColormap.from_list("mpgdre", cmapMPGDre_, cmapMPGDre_.shape[0])
cmapMPG2White = LinearSegmentedColormap.from_list("mpg2white", mpg2white, mpg2white.shape[0])
from matplotlib.cm import register_cmap
register_cmap(cmap=cmapMPGDre)
register_cmap(cmap=cmapMPG2White)
