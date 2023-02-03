import sys
import numpy as np
from tensorflow.keras.models import Model, load_model

def get_patches_from_WSI(path_to_svs):
    '''This should be your usual method
    to get patches from the whole slide image.
    I preffer to use openslide library (https://openslide.org/download/):
    
    import openslide
    
    svs = openslide.OpenSlide(path_to_svs)
    patch = svs.read_region(location=[x,y], size=[width,height], level=0)
    
    '''
    
    pass

def assemble_whole_slide_mask(preds_collection, path_to_save):
    '''This should be your usual method
    to reassemble and save the whole slide mask from predictions.
    I preffer to use memmory mapped bigtif file with tifffile library.                
        
    '''
    
    pass


model = load_model('./models/M-net-fibers-v1.h5',
                   custom_objects = {'mean_iou' : None})

# model.summary()
# sys.exit()

path_to_svs = '/path/to/whole_slide_image.svs'
path_to_save = '/path/to/your/saved_mask.tif'

patch_collection = get_patches_from_WSI(path_to_svs)
preds_collection = []

for patch in patch_collection:
    pred = model.predict(patch, verbose = 0)
    pred = (np.squeeze(pred) > .5) * 1.
    preds_collection.append(pred)

    
assemble_whole_slide_mask(preds_collection, path_to_save)
