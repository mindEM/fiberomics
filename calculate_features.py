import numpy as np
import fiber_methods as fm
from skimage import io

im = io.imread('./demo_images/demo_mask.png')

'''feature extraction is intended for a 2D images - 
i.e. collagen and reticulin fibers should be analysed separately.'''


orientation_f = fm.hog_features(im[:,:,0])
morphometry_f = fm.morphometry_features(im[:,:,0])
texture_f = fm.texture_features(im[:,:,0])
fractal_f = fm.fractal_features(im[:,:,0])

