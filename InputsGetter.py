import ImageProcessor as ip
import numpy as np
import os
from timeit import default_timer as timer
path = 'D:\Pobrane\SIECI_Baza_Znakow\GTSRB\Final_Training'

inputs = np.ones((1024))

start = timer()

for root, directories, filenames in os.walk(path):
    for filename in filenames: 
        vector = ip.get_normalized_vector((os.path.join(root,filename)))
        inputs = np.vstack((inputs, vector))

end = timer()

print("Loading images time: %s seconds" %(end - start))