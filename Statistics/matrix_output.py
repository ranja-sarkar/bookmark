

import pandas as pd
import numpy as np

def res_matrix(file):
    ''' This function outputs the value matrix from a 3D table given Z as the 3rd dimension'''

    df = pd.read_csv(file) #applies to excel or other formats
    n = len(df)
    mapping = {(x, y): z for (x, y, z) in df[["X", "Y", "Z"]].values}
    mat = np.zeros((n, n))
    for i, x in np.ndenumerate(df['X']):
        for j, y in np.ndenumerate(df['Y']):
            mat[j, i] = mapping.get((x, y), 0)  

    return print( mat)


res_matrix('test.csv')
