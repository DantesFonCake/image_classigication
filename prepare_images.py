import numpy as np
from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, exists
files_prefix = "numpy_bitmap"
data_files = [f for f in listdir(files_prefix) if isfile(join(files_prefix, f))]

target = "data-img"
for data_file in data_files:
    directory = data_file[:-4]
    directory = join(target, directory)
    if not exists(directory):
        makedirs(directory)
    
    data = np.load(join(files_prefix, data_file))
    for (i, img)  in enumerate(data):
        new_path = join(directory, str(i))
        new_path = f'{new_path}.png'
        if isfile(new_path):
            continue
        img = Image.fromarray(np.reshape(img, (28, 28)))
        img.save(new_path)
    
    print(f'{directory} completed')
