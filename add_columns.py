"""
Here we add to collected_results.csv new columns that:
a) normalize the overhang depth dnorm=d/2 and height hnorm=h/0.5
b) represent the ratios d/h, h/d (taking care not to divide by zero!)
c) represent sine and cosine of the overhang view angle towards the sky:
   d/sqrt(d^2+h^2) and h/sqrt(d^2+h^2)
d) add the area and the diagonal of the rectangle
   between the overhang and the top of the window:
   dh and sqrt(d^2 + h^2)/sqrt{2^2 + 0.5^2}
e) represent six columns of zeros
   that will be used to fill up the slots expected by DNNs with 8 inputs
   in the case when we test out only the 2 original inputs (depth and height)
"""
import pandas as pd
import numpy as np

df = pd.read_csv('collected_results.csv')

# a) normalize depth and height
df['dnorm'] = df['depth']/2.0
df['hnorm'] = df['height']/0.5

# d) rectangle between the overhang and the top of the window
df['diagonal'] = np.sqrt(df['depth']**2 + df['height']**2)
df['diagnorm'] = df['diagonal']/2.06155         # = sqrt(2^2+0.5^2)
df['area'] = df['depth'] * df['height']

# c) sine and cosine of the overhang view angle towards the sky
df['sine'] = df['height'] / df['diagonal']
df['cosine'] = df['depth'] / df['diagonal']

# b) ratios of depth and height
#    (replacing inf, obtained after division by zero, with 100 times the numerator)
df['d/h'] = df['depth'] / df['height'].where(df['height']>0, other=0.01)
df['h/d'] = df['height'] / df['depth'].where(df['depth']>0, other=0.01)

# e) six columns of zeros
df['zeros1'] = 0
df['zeros2'] = 0
df['zeros3'] = 0
df['zeros4'] = 0
df['zeros5'] = 0
df['zeros6'] = 0

# save everything to the new file
df.to_csv('collected_results_2.csv', index=False, float_format='%g')
