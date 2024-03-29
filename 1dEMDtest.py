from cv2 import *
import numpy as np

# Initialize a and b numpy arrays with coordinates and weights
a = np.zeros((5,2))

for i in range(0,5):
    a[i][1] = i+1

a[0][0] = 1
a[1][0] = 1
a[2][0] = 0
a[3][0] = 0
a[4][0] = 1

b = np.zeros((4,2))

for i in range(0,4):
    b[i][1] = i+1

b[0][0] = 0
b[1][0] = 1
b[2][0] = 0
b[3][0] = 1    

# Convert from numpy array to CV_32FC1 Mat
a64 = cv.fromarray(a)
a32 = cv.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
cv.Convert(a64, a32)

b64 = cv.fromarray(b)
b32 = cv.CreateMat(b64.rows, b64.cols, cv.CV_32FC1)
cv.Convert(b64, b32)

# Calculate Earth Mover's
print cv.CalcEMD2(a32,b32,cv.CV_DIST_L2)
