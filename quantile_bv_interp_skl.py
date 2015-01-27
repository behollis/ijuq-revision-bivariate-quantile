#!/usr/bin/python
#from netCDF4 import * 
import netCDF4
import sys, struct
#import rpy2.robjects as robjects
import random
import math as pm
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import math
#import gaussian_fit
#import sum_of_gaussians_interpolation as sog
from netcdf_reader import *
#from spline_cdf_curve_morphing import *
from mayavi.mlab import *
import mayavi
#from peakfinder import *
from quantile_lerp import *
import os
import datetime 
import time
from cv2 import *
from numpy import *

dt = 'stampeMicroSec_' + str(datetime.datetime.now().microsecond) + '_' 

q_prev_max_vel_x = 0.0
q_prev_max_vel_y = 0.0
e_prev_max_vel_x = 0.0
e_prev_max_vel_y = 0.0
gmm_prev_max_vel_x = 0.0
gmm_prev_max_vel_y = 0.0

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 

SEED_LEVEL = 0
vclin = []

reused_vel_quantile = 0

DEBUG = False

INPUT_DATA_DIR = '/home/behollis/DATA/pierre/ocean/'
#OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/gpDist/'
  
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
OUTPUT_DATA_DIR = '/home/behollis/Dropbox/bvqiPaperCode/'


DEPTH = -2.0
INTEGRATION_DIR = 'b'

ZERO_ARRAY = np.zeros(shape=(MEM,1))

SAMPLES = 600
vclin_x = np.ndarray(shape=(SAMPLES,LAT,LON))
vclin_y = np.ndarray(shape=(SAMPLES,LAT,LON))

vclin_half = np.ndarray(shape=(SAMPLES,LAT/2 + 1,LON/2 + 1,2))
vclin_half = np.ndarray(shape=(SAMPLES,LAT/2 + 1,LON/2 + 1,2))

g_grid_params_array = []
g_grid_kde_array = []
g_grid_quantile_curves_array = []

QUANTILES = 100

def createGlobalParametersArray(dimx, dimy):
    global g_grid_params_array
    
    for idx in range(0,dimx):
        g_grid_params_array.append([])
        for idy in range(0,dimy):
            g_grid_params_array[idx].append([])

def createGlobalKDEArray(dimx, dimy):
    global g_grid_kde_array

    for idx in range(0,dimx):
        g_grid_kde_array.append([])
        for idy in range(0,dimy):
            g_grid_kde_array[idx].append(None)
            
def createGlobalQuantileArray(dimx, dimy):
    global g_grid_quantile_curves_array

    for idx in range(0,dimx):
        g_grid_quantile_curves_array.append([])
        for idy in range(0,dimy):
            g_grid_quantile_curves_array[idx].append([[],[],[]])

#....................... bivariate interp helper functions

def getCoordParts(ppos=[0.0,0.0]):
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    return ppos_parts 

def getGridPoints(ppos=[0.0,0.0], half=False):
    #assume grid points are defined by integer indices
    
    ppos_parts = getCoordParts(ppos)
    
    #print "quantile alpha x: " + str( ppos_parts[0][0] )
    #print "quantile alpha y: " + str( ppos_parts[1][0] )
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    if half == False:
        gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
        gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
        gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
        gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    else:
        gpt0 = [ppos_parts[0][1] / 2, ppos_parts[1][1] / 2]
        gpt1 = [ppos_parts[0][1] / 2 + 1, ppos_parts[1][1] / 2]
        gpt2 = [ppos_parts[0][1] / 2, ppos_parts[1][1] / 2 + 1]
        gpt3 = [ppos_parts[0][1] / 2 + 1, ppos_parts[1][1] / 2 + 1]
    
    return gpt0, gpt1, gpt2, gpt3

def getVclinSamples(gpt0, gpt1, gpt2, gpt3, half=False):
    gpt0_dist = np.zeros(shape=(2,SAMPLES))
    gpt1_dist = np.zeros(shape=(2,SAMPLES))
    gpt2_dist = np.zeros(shape=(2,SAMPLES))
    gpt3_dist = np.zeros(shape=(2,SAMPLES))
    
    for idx in range(0,MEM):#SAMPLES):
        
        if not half:
            gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
            gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
            gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
            gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
            
            gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
            gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
            
            gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
            gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
        else:
            gpt0_dist[0][idx] = vclin_half[idx][gpt0[0]][gpt0[1]][0]
            gpt0_dist[1][idx] = vclin_half[idx][gpt0[0]][gpt0[1]][1]
         
            gpt1_dist[0][idx] = vclin_half[idx][gpt1[0]][gpt1[1]][0]
            gpt1_dist[1][idx] = vclin_half[idx][gpt1[0]][gpt1[1]][1] 
            
            gpt2_dist[0][idx] = vclin_half[idx][gpt2[0]][gpt2[1]][0]
            gpt2_dist[1][idx] = vclin_half[idx][gpt2[0]][gpt2[1]][1] 
            
            gpt3_dist[0][idx] = vclin_half[idx][gpt3[0]][gpt3[1]][0]
            gpt3_dist[1][idx] = vclin_half[idx][gpt3[0]][gpt3[1]][1] 
            
            
   
    return gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist

def plotKDE(var1, var2, kde,distro, mx = (0,0), my = (0, 0), title = '', co = 'green'):
    
    global DIV, DIVR
    
    FIG = plt.figure()

    x_flat = np.r_[np.asarray(distro[0]).min():np.asarray(distro[0]).max():DIV]
    y_flat = np.r_[np.asarray(distro[1]).min():np.asarray(distro[1]).max():DIV]
    #x_flat = np.r_[mx[0]:mx[1]:DIV]
    #y_flat = np.r_[my[0]:my[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(DIVR, DIVR)
    
    AX = FIG.gca(projection='3d')
    AX.set_xlabel(var1)
    AX.set_ylabel(var2)
    AX.set_xlim(x_flat.max(), x_flat.min())
    AX.set_ylim(y_flat.min(), y_flat.max())
    AX.set_zlabel('density')
    AX.set_zlim(0, z.max())
    AX.plot_surface(x, y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True, alpha=1.0, color=co)
    plt.savefig(OUTPUT_DATA_DIR + title + ".jpg")
    #plt.show() 
  
def getKDEGriddata(x_min_max,y_min_max,kde):
    
    global DIV, DIVR
    
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:DIV]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(DIVR, DIVR)
    
    return z
  
 
def interpVelFromEnsemble(ppos=[0.0,0.0]):
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos)
    
    gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist = getVclinSamples(gpt0, gpt1, gpt2, gpt3)
    
    #lerp ensemble samples
    lerp_u_gp0_gp1 = lerp( np.asarray( gpt0_dist[0] ), np.asarray( gpt1_dist[0]), w = ppos_parts[0][0] )
    lerp_u_gp2_gp3 = lerp( np.asarray( gpt2_dist[0] ), np.asarray( gpt3_dist[0]), w = ppos_parts[0][0] ) 
    lerp_u = lerp( np.asarray(lerp_u_gp0_gp1), np.asarray(lerp_u_gp2_gp3), w = ppos_parts[1][0] )  
    
    lerp_v_gp0_gp1 = lerp( np.asarray(gpt0_dist[1] ), np.asarray(gpt1_dist[1]), w = ppos_parts[0][0] )
    lerp_v_gp2_gp3 = lerp( np.asarray(gpt2_dist[1] ), np.asarray(gpt3_dist[1]), w = ppos_parts[0][0] ) 
    lerp_v = lerp( np.asarray(lerp_v_gp0_gp1), np.asarray(lerp_v_gp2_gp3), w = ppos_parts[1][0] )  
    
    #x = linspace( lerp_u[0], lerp_u[-1], len(lerp_u) )
    #y = linspace( lerp_v[0], lerp_v[-1], len(lerp_v) )
    
    x = linspace( -50, 50, 600 )
    y = linspace( -50, 50, 600 )
        
    try:
        k = stats.kde.gaussian_kde((lerp_u,lerp_v)) 
        return k

    except:
        #print "kde not working"
        return None
  

green = '#98fb98'
blue = '#87ceeb'
red = '#f08080'
purple = '#ee82ee'
    


def kl_div_2D_M(mfunc1,mfunc2,min_x=-5, max_x=5, min_y=-5, max_y=5):
    "Calculates the KL divergence D(A||B) between the distributions A and B.\nUsage: div = kl_divergence(A,B)"
    
    global DIV
    
    D = .0

    # Regular grid to evaluate upon
    u_vals = np.r_[min_x:max_x:DIV]
    v_vals = np.r_[min_y:max_y:DIV]
    
    #incr = math.fabs(min_x - max_x) / div
    for u in u_vals:
        for v in v_vals:
            if mfunc2[u,v] != .0 and mfunc1[u,v] != .0:
                D += mfunc1[u,v] * math.log( mfunc1[u,v] / mfunc2[u,v] ) 
    return D 
    
########################################################################################################

#from rpy2.robjects.numpy2ri import numpy2ri
#robjects.conversion.py2ri = numpy2ri
def getVclinSamplesSingle(gpt):
    
    global vclin
    
    gpt0_dist = np.zeros(shape=(2,MEM))
   
    for idx in range(0,MEM):
        gpt0_dist[0][idx] = vclin[idx][gpt[0]][gpt[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt[0]][gpt[1]][SEED_LEVEL][1]
   
    return gpt0_dist

#store in a subsampled array that can be accessed
#with existing interpolation code
def remapGridData():
    for idx in range(0,MEM):
        for lat in range(0,LAT):
            for lon in range(0,LON):
                if lat % 2 == 0 and lon % 2 == 0:
                    #only even coordinate pairs are sampled
                       
                    mlat = 0; mlon = 0
                    
                    if lat == 0:
                        mlat = 0
                    else:
                        mlat = lat / 2
                        
                    if lon == 0:
                        mlon = 0
                    else:
                        mlon = lon / 2
                        
                    #print 'Sampled LAT: ' + str(lat)
                    #print 'Sampled LON: ' + str(lon)
                    #print 'Mapped LAT: ' + str(lat) + ' To: ' + str(mlat)
                    #print 'Sampled LON: ' + str(lon)  + ' To: ' + str(mlon)
                    
                    vclin_half[idx][mlat][mlon][0] = vclin[idx][lat][lon][SEED_LEVEL][0]
                    vclin_half[idx][mlat][mlon][1] = vclin[idx][lat][lon][SEED_LEVEL][1]
        
def loadNetCdfData(var1, var2):
    global vclin
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #realizations reader 
    #rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    netFileReal = netCDF4.Dataset(pe_dif_sep2_98_file)
    
    #central forecasts reader 
    #creader = NetcdfReader(pe_fct_aug25_sep2_file)
    netFileCent = netCDF4.Dataset(pe_fct_aug25_sep2_file)
    
    SCALE = 10.0
    
    temp8 = np.expand_dims(netFileCent.variables[ var2 ][ 7 ], axis=3)
    salt8 = np.expand_dims(SCALE*netFileReal.variables[ var1 ][ 7 ], axis=3)
    
    temp = np.expand_dims(netFileReal.variables[var2][:], axis = 4)
    salt = np.expand_dims(SCALE*netFileReal.variables[var1][:], axis = 4)
   
    #vclin8 = creader.readVarArray('vclin', 7)
    #vclin8 = creader.readVarArray('vclin', 7)
    
    #deviations from central forecast for all 600 realizations
    #vclin = rreader.readVarArray('vclin')  
    
    vclin = np.concatenate((salt, temp), axis=4)
    vclin8 = np.concatenate((salt8, temp8), axis=3)
    
    vclin = addCentralForecast(vclin, vclin8, level_start=SEED_LEVEL, level_end=SEED_LEVEL)  
    
        
#import rpy2.robjects.numpy2ri as rpyn

###### Quantile interp code ########
def bilinearBivarQuantLerp(f1, f2, f3, f4, x1, y1, x2, y2, x3, y3, x4, y4, alpha, beta):
    qstart = time.clock()
    a0 = 1.0 - alpha
    b0 = alpha
    a1 = 1.0 - beta
    b1 = beta
    
    try:
        f_one = f1((x1,y1))
        f_two = f2((x2,y2))
        f_three = f3((x3,y3))
        f_four = f4((x4,y4))            
        
        f_bar_0 = f_one * f_two / (a0*f_two + b0*f_one) 
        f_bar_1 = f_three * f_four / (a0*f_four + b0*f_three) 
        
        f_bar_01 = f_bar_0 * f_bar_1 / (a1*f_bar_1 + b1*f_bar_0)
    except:
        #print 'problem with calculated interpolant z value...'
        f_bar_01[0] = -1 #failed
    
    qend = time.clock()
    totTime = qend-qstart
    
    return f_bar_01[0], totTime

DIV = 200j 
DIVR = 200

def findBivariateQuantilesSinglePass(kde,arr):
    
    #cpu time
    print 'integrating KDE...'
    qstart = time.clock() 
    
    global QUANTILES, DIV, DIVR
    
    
    u_min = arr.T[:,0].min()
    u_max = arr.T[:,0].max()
    v_min = arr.T[:,1].min()
    v_max = arr.T[:,1].max()
    
    # tolearance of cdf value, not a function of integration grid
    TOL = (1./QUANTILES) / 2. #np.max([((u_max - u_min) / divs), ((v_max - v_min) / divs)])
    
    u_extent = math.fabs( u_min - u_max )
    v_extent = math.fabs( v_min - v_max )
    
    incr_x = u_extent / DIVR#div_x 
    incr_y = v_extent / DIVR#div_y
    x_div = np.r_[u_min:u_max:DIV]
    y_div = np.r_[v_min:v_max:DIV]
    
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #integrate kde to find bivariate ecdf
    qs = list(spread(0.0, 1.0, QUANTILES-1, mode=3)) 
    qs.sort()
    
    qcurvex = []
    qcurvey = []
    for q in qs:
        qcurvex.append([])
        qcurvey.append([])
    
    for x in x_div:
        cd = 0.0
        
        for y in y_div:
            #print y
            low_bounds = (u_min,v_min)
            high_bounds = (x+incr_x,y+incr_y)
            
            cd = kde.integrate_box(low_bounds, high_bounds, maxpts=None)
            
            for idx, q in enumerate(qs):
                if cd <= q + TOL and cd >= q - TOL:
                    #print "gathering points for quantile curve #: " + + str(idx) + " out of " + str(QUANTILES)
                    qcurvex[idx].append(x)
                    qcurvey[idx].append(y)
            
            z_pos.append(cd)
            x_pos.append(x)
            y_pos.append(y)
            
    #print 'finished computing quantile curves'
    
    #cpu time
    qend = time.clock()
    qtot = qend - qstart
    print 'cpu time for KDE integration (EDCF): ' + str(qtot) 
            
    return x_pos, y_pos, z_pos, qcurvex, qcurvey

MID_RANGE_QUANTILE_CURVE_POINTS = 150

def lerpBivariate3(gp0, gp1, gp2, gp3, alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3, arr, use_cache=False):
    
    totalCpuStart = time.clock()
    
    global g_grid_quantile_curves_array, QUANTILES, MID_RANGE_QUANTILE_CURVE_POINTS

    degree = 3;smoothing = None
    INTERP = 'linear'
    #spline_curve0=[];spline_curve1=[];spline_curve2=[];spline_curve3=[]
    
    qcurveCpu = 0.
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex0 = gp0_qcurve[0]
    qcurvey0 = gp0_qcurve[1]
    spline_curve0 = gp0_qcurve[2]
    if len(gp0_qcurve[0]) == 0 or use_cache == False:
        #print 'computing quantile curves gp0...'
        x_pos0, y_pos0, z_pos0, qcurvex0, qcurvey0 = findBivariateQuantilesSinglePass(gp0,arr[0])
        #plotXYZScatterQuants(qcurvex0, qcurvey0, title='qcurve0')
        spline_curve0 = []
        
        qstart = time.clock() 
        for q in range(0,len(qcurvex0)):
            if len(qcurvex0[q]) > degree: #must be greater than k value
                #spline_curve0.append(interpolate.UnivariateSpline(qcurvex0[q], qcurvey0[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve0.append(interpolate.interp1d(qcurvex0[q], qcurvey0[q], kind=INTERP))
            else:
                spline_curve0.append([None])
        qend = time.clock()
        qcurveCpu += qend - qstart
        
        print 'Quantile curve parameterization: ' + str(qcurveCpu)
        
        g_grid_quantile_curves_array[i][j][0] = qcurvex0
        g_grid_quantile_curves_array[i][j][1] = qcurvey0
        g_grid_quantile_curves_array[i][j][2] = spline_curve0
        
    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex1 = gp1_qcurve[0]
    qcurvey1 = gp1_qcurve[1]
    spline_curve1 = gp1_qcurve[2]
    if len(gp1_qcurve[0]) == 0 or use_cache == False:
        #print 'computing quantile curves gp1...'
        x_pos1, y_pos1, z_pos1, qcurvex1, qcurvey1 = findBivariateQuantilesSinglePass(gp1,arr[1])
        #plotXYZScatterQuants(qcurvex1, qcurvey1, title='qcurve1')
        spline_curve1 = []
        for q in range(0,len(qcurvex1)):
            if len(qcurvex1[q]) > degree:    
                #spline_curve1.append(interpolate.UnivariateSpline(qcurvex1[q], qcurvey1[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve1.append(interpolate.interp1d(qcurvex1[q], qcurvey1[q], kind=INTERP))
            else:
                spline_curve1.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex1
        g_grid_quantile_curves_array[i][j][1] = qcurvey1
        g_grid_quantile_curves_array[i][j][2] = spline_curve1
        
    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex2 = gp2_qcurve[0]
    qcurvey2 = gp2_qcurve[1]
    spline_curve2 = gp2_qcurve[2]
    if len(gp2_qcurve[0]) == 0 or use_cache == False:
        #print 'computing quantile curves gp2...'
        x_pos2, y_pos2, z_pos2, qcurvex2, qcurvey2 = findBivariateQuantilesSinglePass(gp2,arr[2])
        #plotXYZScatterQuants(qcurvex2, qcurvey2, title='qcurve2')
        spline_curve2 = []
        for q in range(0,len(qcurvex2)):
            if len(qcurvex2[q]) > degree:
                #spline_curve2.append(interpolate.UnivariateSpline(qcurvex2[q], qcurvey2[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve2.append(interpolate.interp1d(qcurvex2[q], qcurvey2[q], kind=INTERP))
            else:
                spline_curve2.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex2
        g_grid_quantile_curves_array[i][j][1] = qcurvey2
        g_grid_quantile_curves_array[i][j][2] = spline_curve2
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex3 = gp3_qcurve[0]
    qcurvey3 = gp3_qcurve[1]
    spline_curve3 = gp3_qcurve[2]
    if len(gp3_qcurve[0]) == 0 or use_cache == False:
        #print 'computing quantile curves gp3...'
        x_pos3, y_pos3, z_pos3, qcurvex3, qcurvey3 = findBivariateQuantilesSinglePass(gp3, arr[3])
        #plotXYZScatterQuants(qcurvex3, qcurvey3, title='qcurve3')
        spline_curve3 = []
        for q in range(0,len(qcurvex3)):
            if len(qcurvex3[q]) > degree:
                #spline_curve3.append(interpolate.UnivariateSpline(qcurvex3[q], qcurvey3[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve3.append(interpolate.interp1d(qcurvex3[q], qcurvey3[q], kind=INTERP))
            else:
                spline_curve3.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex3
        g_grid_quantile_curves_array[i][j][1] = qcurvey3
        g_grid_quantile_curves_array[i][j][2] = spline_curve3
        
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #smaller quantiles have longer quantile curves, so we adjust this number based on quantile below
    num_pts_to_eval_on_curve = MID_RANGE_QUANTILE_CURVE_POINTS
    
    pdfEvalsCpu = 0.
    qCurveInterpCpu = 0.
    qstartCurv = time.clock()
    for iq in range(0,QUANTILES):#, q in enumerate(qcurvex0):
        
        #if iq <= 0.5*QUANTILES:
        #    num_pts_to_eval_on_curve = 3*MID_RANGE_QUANTILE_CURVE_POINTS
        #else:
        #    num_pts_to_eval_on_curve = MID_RANGE_QUANTILE_CURVE_POINTS
            
        #print str(iq) + "th quantile curve being lerped out of " + str(QUANTILES)
        #get an x,y pair for current quantile on each pdf end points
        #limit = min([len(qcurvex0[iq]), len(qcurvex1[iq]), len(qcurvex2[iq]), len(qcurvex3[iq])])
        #for idx in range(0,limit):
        epts0 = [];epts1 = [];epts2=[];epts3=[]
        cur_y0_parametrized_pts=[];cur_y1_parametrized_pts=[];cur_y2_parametrized_pts=[];cur_y3_parametrized_pts=[]
        if spline_curve0[iq] != None and spline_curve1[iq] != None \
            and spline_curve2[iq] != None and spline_curve3[iq] != None and len(qcurvex0[iq]) > degree and \
            len(qcurvex1[iq]) > degree and len(qcurvex2[iq]) > degree and len(qcurvex3[iq]) > degree:
            #if iq < int(0.25*QUANTILES):
            #    num_pts_to_eval_on_curve = int(2*MID_RANGE_QUANTILE_CURVE_POINTS )
            #elif iq > int(0.75*QUANTILES):
            #    num_pts_to_eval_on_curve = int(0.5*MID_RANGE_QUANTILE_CURVE_POINTS )
            #print '    evaluating spline...'
            epts0 = linspace(qcurvex0[iq][0], qcurvex0[iq][-1], num_pts_to_eval_on_curve)
            epts1 = linspace(qcurvex1[iq][0], qcurvex1[iq][-1], num_pts_to_eval_on_curve)
            epts2 = linspace(qcurvex2[iq][0], qcurvex2[iq][-1], num_pts_to_eval_on_curve)
            epts3 = linspace(qcurvex3[iq][0], qcurvex3[iq][-1], num_pts_to_eval_on_curve) 
            cur_y0_parametrized_pts = spline_curve0[iq](epts0) 
            cur_y1_parametrized_pts = spline_curve1[iq](epts1)
            cur_y2_parametrized_pts = spline_curve2[iq](epts2)
            cur_y3_parametrized_pts = spline_curve3[iq](epts3)
            #print '...finished evaluating spline!'
        else:
            continue
        
        
        
        for idx in range(0,num_pts_to_eval_on_curve): #evaluate points along each parameterized quantile curve
            #print '    lerping point: ' +str(idx)+ ' out of ' + str(num_pts_to_eval_on_curve)
            cur_x0 = epts0[idx]#qcurvex0[iq][idx]
            cur_y0 = cur_y0_parametrized_pts[idx]#qcurvey0[iq][idx]
            cur_x1 = epts1[idx]#qcurvex1[iq][idx]
            cur_y1 = cur_y1_parametrized_pts[idx]#qcurvey1[iq][idx]
            cur_x2 = epts2[idx]#qcurvex2[iq][idx]
            cur_y2 = cur_y2_parametrized_pts[idx]#qcurvey2[iq][idx]
            cur_x3 = epts3[idx]#qcurvex3[iq][idx]
            cur_y3 = cur_y3_parametrized_pts[idx]#qcurvey3[iq][idx]
            
            dir_vec0 = np.asarray([cur_x1, cur_y1]) - np.asarray([cur_x0, cur_y0])
            dir_vec1 = np.asarray([cur_x3, cur_y3]) - np.asarray([cur_x2, cur_y2])
            
            interpolant_xy0 = np.asarray([cur_x0,cur_y0]) + alpha_x * dir_vec0
            interpolant_xy1 = np.asarray([cur_x2,cur_y2]) + alpha_x * dir_vec1
            
            dir_vec_bar = np.asarray([interpolant_xy1[0],interpolant_xy1[1]]) - \
                np.asarray([interpolant_xy0[0],interpolant_xy0[1]])
            
            interpolant_xy_bar = np.asarray([interpolant_xy0[0],interpolant_xy0[1]]) + alpha_y * dir_vec_bar
            
            qend = time.clock()
            qCurveInterpCpu += qend - qstartCurv 
            #print iq
            #print qCurveInterpCpu
            
            z, cpuTime = bilinearBivarQuantLerp(gp0, gp1, gp2, gp3, cur_x0, cur_y0, \
                                        cur_x1, cur_y1, cur_x2, cur_y2, cur_x3, cur_y3, \
                                         alpha_x, alpha_y)
            
            pdfEvalsCpu += cpuTime
            
            qstartCurv = time.clock()
            
            if z < 0. or math.isnan(z) or math.isinf(z):
            #    print 'WARNING: interpolated pdf has NaN, inf or negative result...skipping data point.'
                continue
           
            x_pos.append(interpolant_xy_bar[0])
            y_pos.append(interpolant_xy_bar[1])
            z_pos.append(z)  
            #print '        lerping point between quantile curves: ' + str(iq) + ' was successful!'
            
    #print 'finished lerping all quantile curve for interpolant distro...'
    
   
    totalCpu = time.clock() - totalCpuStart
    
    print 'Quantile curve interpolation: ' + str(qCurveInterpCpu)
    print 'PDF surface pts. eval: ' + str(pdfEvalsCpu)
    print 'Total Cpu for interp: ' + str(totalCpu)
    return x_pos, y_pos, z_pos

def interpFromQuantiles3(ppos=[0.0,0.0], number=0,sl=0, ignore_cache = 'False', half=False):
    
    #print ppos
    
    global g_grid_kde_array
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos, half)
    gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp = getVclinSamples(gpt0, gpt1, gpt2, gpt3, half)
    
    samples_arr = [gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp]
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_kde = g_grid_kde_array[i][j]
    if gp0_kde is None or ignore_cache == 'True':
        try:
            gp0_kde = stats.kde.gaussian_kde( ( gpt0_samp[0][:], gpt0_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
           
        g_grid_kde_array[i][j] = gp0_kde

    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_kde = g_grid_kde_array[i][j]
    if gp1_kde is None or ignore_cache == 'True':
        try:
            gp1_kde = stats.kde.gaussian_kde( ( gpt1_samp[0][:], gpt1_samp[1][:] ) )
        
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
        
        g_grid_kde_array[i][j] = gp1_kde

    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_kde = g_grid_kde_array[i][j]
    if gp2_kde is None or ignore_cache == 'True':
        try:
            gp2_kde = stats.kde.gaussian_kde( ( gpt2_samp[0][:], gpt2_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp2_kde
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_kde = g_grid_kde_array[i][j]
    if gp3_kde is None or ignore_cache == 'True':
        try:
            gp3_kde = stats.kde.gaussian_kde( ( gpt3_samp[0][:], gpt3_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp3_kde

    #interp dist for x dim  
    alpha_x = 0.
    alpha_y = 0.
    if half == False:
        alpha_x = ppos_parts[0][0]
        alpha_y = ppos_parts[1][0]
    else:
        #alpha_x = pm.modf(ppos_parts[0][1] / 2)[0]
        #alpha_y = pm.modf(ppos_parts[1][1] / 2)[0]
        alpha_x = pm.modf(ppos[0] / 2)[0]
        alpha_y = pm.modf(ppos[1] / 2)[0]
    
    x3, y3, z3 = lerpBivariate3(gp0_kde, gp1_kde, gp2_kde, gp3_kde,\
                                            alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3, samples_arr, use_cache=half )
    
    '''
    plotKDE(gp0_kde,gpt0_samp, title="gpt0_kde", co = green)
    plotKDE(gp1_kde,gpt1_samp, title="gpt1_kde", co = green)
    plotKDE(gp2_kde,gpt2_samp, title="gpt2_kde", co = green)
    plotKDE(gp3_kde,gpt3_samp, title="gpt3_kde", co = green)
    '''
    
    return samples_arr, [x3,y3,z3]

def computeDistroFunction(x_pos,y_pos,z_pos, mmx, mmy):
    
    global DIV, DIVR
   
    X = np.r_[mmx[0]:mmx[1]:DIV]
    Y = np.r_[mmy[0]:mmy[1]:DIV]
    X,Y = np.meshgrid(X,Y)
    
    pts = np.append(np.asarray(x_pos).reshape(-1,1),np.asarray(y_pos).reshape(-1,1),axis=1)
    
    
    distro_eval = None
    success = False
    interp_type = ''
    
    #print "in computeDistroFunction..."
    
    while success is False:
        try:
            distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='linear', fill_value=0.0)
            success = True
            interp_type = 'linear'
        except:
            try:
                distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='cubic', fill_value=0.0)
                success = True
                interp_type = 'cubic'
            except:
                continue

    #print "finished distro function comp..."
    #print 'success? -> ' + str(success)
    #print 'type? -> ' + str(interp_type)
    return distro_eval, interp_type, success

def plotXYZScatter(var1, var2, (u_min,u_max),(v_min, v_max), x_pos,y_pos,z_pos, title = '', arr=[]):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(list(x_pos), list(y_pos), list(z_pos),s=0.05)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    
    ax.set_xlim(u_max, u_min)
    ax.set_ylim(v_min, v_max)
        
    ax.set_zlabel('density')
    ax.set_zlim(0, np.asarray(z_pos).max())
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "scatter.png")
    #plt.show()
    
def plotXYZSurf(var1, var2, mmx, mmy, surface, title = '', arr=[], col='0.75'):
    
    global DIV, DIVR
    
    FIG = plt.figure()
    
    x_flat = np.r_[mmx[0]:mmx[1]:DIV]
    y_flat = np.r_[mmy[0]:mmy[1]:DIV]
    X,Y = np.meshgrid(x_flat,y_flat)
    
    AX = FIG.gca(projection='3d')
    AX.set_xlabel(var1)
    AX.set_ylabel(var2)
    AX.set_xlim(x_flat.max(), x_flat.min())
    AX.set_ylim(y_flat.min(), y_flat.max())
    AX.set_zlabel('density')
    AX.set_zlim(0, surface.max())
    qstart = time.clock()
    AX.plot_surface(X, Y, surface, rstride=2, cstride=2, alpha=1.0, \
                    linewidth=0.1, antialiased=True, color=col)
    qend = time.clock()
    tot = qend - qstart
    print 'PDF linear surface interpolation: ' + str(tot)
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + ".jpg")
    #plt.show() 
    
def convertNumpyArrayToOpenCVSignature(numpyMat):
    #append coord of each row for opencv conversion.
    newMat = np.zeros( shape = ( numpyMat.shape[0] * numpyMat.shape[1], 1+ len(numpyMat.shape) ) )
                                       
    for row in range( 0, numpyMat.shape[0], 1 ): 
        for col in range( 0, numpyMat.shape[1], 1 ): 
            newRow = row*col + col
            newMat[newRow][0] = numpyMat[row][col]
            newMat[newRow][1] = row
            newMat[newRow][2] = col
            
    # Convert from numpy array to CV_32FC1 Mat
    a64 = cv.fromarray(newMat)
    a32 = cv.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
    cv.Convert(a64, a32)

    return a32
    
def main():
    
    var1 = 'salt'
    var2 = 'temp'
    
    #var1 = 'NO3'
    #var2 = 'temp'
    
    loadNetCdfData(var1, var2)
    remapGridData()
    
    createGlobalKDEArray(LAT,LON)
    createGlobalQuantileArray(LAT,LON)
    
    #ppos = [0,36]
    ppos = [44,30]
    
    kdes = []
    
    ''' http://www.tayloredmktg.com/rgb/#PA '''
    green = '#98fb98'
    blue = '#87ceeb'
    red = '#f08080'
    purple = '#ee82ee'
    
    #DPER = 0.0 # to avoid cut off in surface interp
      
    for idx in range(0,3):
        xpos = ppos[0] + float(1.0*idx)
        
        #find KDE benchmark
        distro = getVclinSamplesSingle([xpos,ppos[1]])
        
        kde = stats.kde.gaussian_kde(distro)
        x_min = np.asarray(distro[0]).min() 
        x_max = np.asarray(distro[0]).max()
        y_min = np.asarray(distro[1]).min()
        y_max = np.asarray(distro[1]).max()
        
        #x_min -= DPER*x_min
        #x_max += DPER*x_max
        #y_min -= DPER*y_min
        #y_max += DPER*y_max
        
        mfunc1 = getKDEGriddata((x_min,x_max), (y_min,y_max), kde)

        #http://stackoverflow.com/questions/15706339/how-to-compute-emd-for-2-numpy-arrays-i-e-histogram-using-opencv
        
        
        sigKDE = convertNumpyArrayToOpenCVSignature(mfunc1)
        emdKDE = cv.CalcEMD2(sigKDE, sigKDE, cv.CV_DIST_L2)
        
        samples_arr, evalfunc = interpFromQuantiles3(ppos=[xpos,ppos[1]], ignore_cache = 'True', half=True)
        
        x_min = np.asarray(evalfunc[0]).min() 
        x_max = np.asarray(evalfunc[0]).max()
        y_min = np.asarray(evalfunc[1]).min()
        y_max = np.asarray(evalfunc[1]).max()
        
        #x_min -= DPER*x_min
        #x_max += DPER*x_max
        #y_min -= DPER*y_min
        #y_max += DPER*y_max
        
        distro2, interpType, suc = computeDistroFunction(evalfunc[0],evalfunc[1],evalfunc[2], \
                                                         (x_min,x_max), (y_min,y_max))
        
        sigQuant = convertNumpyArrayToOpenCVSignature(distro2)
        emdQuant = cv.CalcEMD2(sigKDE, sigQuant, cv.CV_DIST_L2)
        
        titleKDE = dt + str(xpos) + '_' + str(ppos[1]) + '_kde_emd_' + str(emdKDE) 
        titleQuantInterp = dt + str(xpos) + '_' + str(ppos[1]) + '_q_emd_' + str(emdQuant) 
        
        #plot interpolants in their range
        plotKDE(var1, var2, kde,distro,(x_min,x_max), (y_min,y_max), titleKDE, co=green)
        plotXYZSurf(var1, var2, (x_min,x_max), (y_min,y_max), distro2, titleQuantInterp, samples_arr, col=blue)
        plotXYZScatter(var1, var2, (x_min,x_max), (y_min,y_max), evalfunc[0],evalfunc[1],evalfunc[2], title=dt + str(xpos) + \
                       '_' + str(ppos[1]) + '_Interp_', arr=samples_arr )
             
        #find full resolution, non-interpolated distribution       
        if xpos == 45.0:
            #find quantile approx (include surface interpolant choice)
            
            x_min = np.asarray(distro[0]).min()
            x_max = np.asarray(distro[0]).max()
            y_min = np.asarray(distro[1]).min()
            y_max = np.asarray(distro[1]).max()
            
            #x_min -= DPER*x_min
            #x_max += DPER*x_max
            #y_min -= DPER*y_min
            #y_max += DPER*y_max
            
            samples_arr_a, evalfunc_a = interpFromQuantiles3(ppos=[xpos,ppos[1]], ignore_cache = 'True', half=False)
            
            distro2_a, interpType_a, suc = computeDistroFunction(evalfunc_a[0],evalfunc_a[1],evalfunc_a[2], \
                                                             (x_min,x_max), (y_min,y_max))
            
            actualQuantSig = convertNumpyArrayToOpenCVSignature(distro2_a)
            emdQuantA = cv.CalcEMD2(sigKDE, actualQuantSig, cv.CV_DIST_L2)
            
            title4_a = dt + str(xpos) + '_' + str(ppos[1]) + '_q_not_interpolated_emd_' + str(emdQuantA)
            
            plotXYZSurf(var1, var2, (x_min,x_max), (y_min,y_max), distro2_a, title4_a, samples_arr_a, col=blue)
            plotXYZScatter(var1, var2, (x_min,x_max), (y_min,y_max), evalfunc_a[0],evalfunc_a[1],evalfunc_a[2], title=dt + str(xpos) + \
                           '_' + str(ppos[1]) + '_NOT_Interp_', arr=samples_arr_a )
    

    print 'finished!'
            
if __name__ == "__main__":  
    main()
