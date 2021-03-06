
import numpy as np
arr = [[335, 209, 214, 514, 220, 222, 506, 226, 228, 503],
       [0, 339, 336, 334, 333, 332, 331, 330, 328, 326],
       [335, 390, 388, 386, 385, 382, 381, 380, 378, 377],
       [0, 388, 385, 384, 383, 382, 381, 377, 376, 373],
       [0, 336, 334, 331, 330, 328, 324, 321, 317, 315],
       [0, 180, 493, 185, 188, 189, 489, 195, 196, 197],
       [619, 251, 373, 288, 603, 146, 147, 600, 599, 154]]
arr = np.asarray(arr, float)
normed = (arr - arr.mean(axis=0))/(arr.std(axis=0)+(10**-6))
print(normed)
print(normed.std(axis = 0))