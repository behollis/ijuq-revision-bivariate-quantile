'''
Computing Spearman test for correlation on Ocean data.
'''

import sys
import netcdf_reader
import glob
import numpy as np
import matplotlib.pyplot as plt
from mayavi.mlab import *
import netCDF4 
from scipy import *
from scipy.stats.vonmises_cython import scipy

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 

OUTPUT_DATA_DIR = '/home/behollis/Dropbox/bvqiPaperCode'
INPUT_DATA_DIR = '/home/behollis/DATA/pierre/ocean/'
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'

'''
behollis@behollis-HPE-480t:~/DATA/pierre/ocean$ ncdump -h pe_fct_aug25_sep2.nc
netcdf pe_fct_aug25_sep2 {
dimensions:
        tlat = 90 ;
        tlon = 53 ;
        vlat = 90 ;
        vlon = 53 ;
        level = 16 ;
        outlev = 16 ;
        vector = 2 ;
        time = UNLIMITED ; // (9 currently)
        axis2 = 2 ;
        axis3 = 3 ;
        tslev = 18 ;
        usrdim1 = 10 ;
        usrdim2 = 30 ;
        len24 = 24 ;
        len80 = 80 ;
variables:
        char runpe(len24) ;
        char titlrun(len80) ;
        char titlmdl(len80) ;
        char titleic(len80) ;
        char titlebc(len80) ;
        char runic(len24) ;
        int imt ;
                imt:long_name = "number of tracer points in the x-direction" ;
        int jmt ;
                jmt:long_name = "number of tracer points in the y-direction" ;
        int km ;
                km:long_name = "number of vertical levels" ;
        int nt ;
                nt:long_name = "number of tracer type variables" ;
        int lseg ;
                lseg:long_name = "maximum number of sets of Start and
End indices" ;
        int misle ;
                misle:long_name = "maximum number of islands in the
model basin" ;
        int lbc ;
                lbc:long_name = "number of arrays of slab incidental data" ;
        int nfirst ;
                nfirst:long_name = "start/restart switch" ;
                nfirst:options = "0: restart, 1: start from scratch" ;
        int nlast ;
                nlast:long_name = "number of timesteps to calculate" ;
        int nnergy ;
                nnergy:long_name = "number of timesteps between energy
diagnostics" ;
        int ntsout ;
                ntsout:long_name = "number of timesteps between output
of data" ;
                ntsout:valid_min = 1 ;
        int ntsi ;
                ntsi:long_name = "number of timesteps between print of
information" ;
                ntsi:valid_min = 1 ;
        int nmix ;
                nmix:long_name = "number of timesteps between mixing
timesteps" ;
        int mxscan ;
                mxscan:long_name = "maximum number of iterations for
relaxation" ;
        int mixvel ;
                mixvel:long_name = "momentum horizontal mixing scheme" ;
                mixvel:options = "1: Shapiro, 2: Laplacian" ;
        int mixtrc ;
                mixtrc:long_name = "tracers horizontal mixing scheme" ;
                mixtrc:options = "1: Shapiro, 2: Laplacian" ;
        int mixztd ;
                mixztd:long_name = "barotropic vorticity horizontal
mixing scheme" ;
                mixztd:options = "1: Shapiro, 2: Laplacian" ;
        float am ;
                am:long_name = "horizontal mixing of momentum" ;
                am:units = "centimeter2 second-1" ;
        float ah ;
                ah:long_name = "horizontal mixing of heat, salinity
and tracers" ;
                ah:units = "centimeter2 second-1" ;
        float fkpm ;
                fkpm:long_name = "vertical mixing of momentum" ;
                fkpm:units = "centimeter2 second-1" ;
        float fkph ;
                fkph:long_name = "vertical mixing of heat, salinity
and tracers" ;
                fkph:units = "centimeter2 second-1" ;
        int nordv ;
                nordv:long_name = "velocity: order of Shapiro filter" ;
                nordv:valid_min = 2 ;
                nordv:valid_max = 8 ;
        int ntimv ;
                ntimv:long_name = "velocity: number of Shapiro filter
applications" ;
                ntimv:valid_min = 1 ;
        int nfrqv ;
                nfrqv:long_name = "velocity: timesteps between Shapiro
filtering" ;
                nfrqv:valid_min = 0 ;
        int nordt ;
                nordt:long_name = "tracers: order of Shapiro filter" ;
                nordt:valid_min = 2 ;
                nordt:valid_max = 8 ;
        int ntimt ;
                ntimt:long_name = "tracers: number of Shapiro filter
applications" ;
                ntimt:valid_min = 1 ;
        int nfrqt ;
                nfrqt:long_name = "tracers: timesteps between Shapiro
filtering" ;
                nfrqt:valid_min = 0 ;
        int nordp ;
                nordp:long_name = "transport: order of Shapiro filter" ;
                nordp:valid_min = 2 ;
                nordp:valid_max = 8 ;
        int ntimp ;
                ntimp:long_name = "transport: number of Shapiro filter
applications" ;
                ntimp:valid_min = 1 ;
        int nfrqp ;
                nfrqp:long_name = "transport: timesteps between
Shapiro filtering" ;
                nfrqp:valid_min = 0 ;
        int nordz ;
                nordz:long_name = "vorticity: order of Shapiro filter" ;
                nordz:valid_min = 2 ;
                nordz:valid_max = 8 ;
        int ntimz ;
                ntimz:long_name = "vorticity: number of Shapiro filter
applications" ;
                ntimz:valid_min = 1 ;
        int nfrqz ;
                nfrqz:long_name = "vorticity: timesteps between
Shapiro filtering" ;
                nfrqz:valid_min = 0 ;
        float dtts ;
                dtts:long_name = "length of timestep on tracers" ;
                dtts:units = "seconds" ;
        float dtuv ;
                dtuv:long_name = "length of timestep on momentum" ;
                dtuv:units = "seconds" ;
        float dtsf ;
                dtsf:long_name = "length of timestep on transport
streamfunction" ;
                dtsf:units = "seconds" ;
        float dstart ;
                dstart:long_name = "initialization time" ;
                dstart:units = "modified Julian day" ;
                dstart:add_offset = 2440000.f ;
        float sor ;
                sor:long_name = "coefficient for over-relaxation" ;
                sor:valid_min = 1.f ;
                sor:valid_max = 2.f ;
        float crit ;
                crit:long_name = "RMS relaxation convergence criterion" ;
                crit:units = "percentage" ;
        float acor ;
                acor:long_name = "coefficient for implicit treatment
of Coriolis term" ;
                acor:valid_min = 0.f ;
                acor:valid_max = 1.f ;
        float rlngd ;
                rlngd:long_name = "domain centroid longitude" ;
                rlngd:units = "degrees_east" ;
        float rlatd ;
                rlatd:long_name = "domain centroid latitude" ;
                rlatd:units = "degrees_north" ;
        float thetad ;
                thetad:long_name = "domain rotation angle" ;
                thetad:units = "degrees" ;
        float gridx ;
                gridx:long_name = "zonal grid spacing across tracer boxes" ;
                gridx:units = "centimeter" ;
        float gridy ;
                gridy:long_name = "meridional grid spacing across
tracer boxes" ;
                gridy:units = "centimeter" ;
        float rho0 ;
                rho0:long_name = "mean density of seawater" ;
                rho0:units = "kilogram meter-3" ;
        float smean ;
                smean:long_name = "mean salinity subtracted during
computations" ;
                smean:units = "PSS" ;
        float zc1 ;
                zc1:long_name = "minimum depth of the coordinate interface" ;
                zc1:units = "centimeter" ;
        float zc2 ;
                zc2:long_name = "maximum depth of the coordinate interface" ;
                zc2:units = "centimeter" ;
        float zref ;
                zref:long_name = "reference depth for coordinate interface" ;
                zref:units = "centimeter" ;
        float zslope ;
                zslope:long_name = "slope parameter of the coordinate
interface" ;
        float attphy ;
                attphy:long_name = "phytoplankton light attenuation scale" ;
                attphy:units = "liter micromole-1 meter-1" ;
        float parfrac ;
                parfrac:long_name = "fraction of shortwave radiation
that is photosynthetically active" ;
        float photorm ;
                photorm:long_name = "maximum photosynthetic rate" ;
                photorm:units = "milligramCarbon
milligramChlorophyll-1 second-1" ;
        float photor0 ;
                photor0:long_name = "initial slope of photosynthesis
response to light" ;
                photor0:units = "milligramCarbon meter2
milligramChlorophyll-1 micromolePhotons-1" ;
        float photoinh ;
                photoinh:long_name = "photoinhibition parameter" ;
                photoinh:units = "milligramCarbon meter2
milligramChlorophyll-1 micromolePhotons-1" ;
        float hsno3 ;
                hsno3:long_name = "half saturation constant for
nitrate uptake" ;
                hsno3:units = "micromole liter-1" ;
        float hsnh4 ;
                hsnh4:long_name = "half saturation constant for
ammonium uptake" ;
                hsnh4:units = "micromole liter-1" ;
        float no3inh ;
                no3inh:long_name = "strength of ammonium inhibition of
nitrate uptake" ;
                no3inh:units = "liter micromole-1" ;
        float phylr1 ;
                phylr1:long_name = "linear phytoplankton mortality rate" ;
                phylr1:units = "day-1" ;
        float phylr2 ;
                phylr2:long_name = "quadratic phytoplankton mortality rate" ;
                phylr2:units = "liter micromole-1 day-1" ;
        float grazrm ;
                grazrm:long_name = "maximum phytoplankton grazing rate" ;
                grazrm:units = "day-1" ;
        float civlev ;
                civlev:long_name = "Ivlev constant for zooplankton
grazing of phytoplankton" ;
                civlev:units = "liter micromole-1" ;
        float zooexcn ;
                zooexcn:long_name = "fraction of zooplankton grazing
that is excreted as ammonium" ;
        float zooexcd ;
                zooexcd:long_name = "fraction of zooplankton grazing
that is excreted as detritus" ;
        float zoolr1 ;
                zoolr1:long_name = "linear zooplankton loss rate" ;
                zoolr1:units = "day-1" ;
        float zoolr2 ;
                zoolr2:long_name = "quadratic zooplankton loss rate" ;
                zoolr2:units = "liter micromole-1 day-1" ;
        float zoolf1 ;
                zoolf1:long_name = "fraction of linear zooplankton
loss to detritus" ;
        float zoolf2 ;
                zoolf2:long_name = "fraction of quadratic zooplankton
loss to detritus" ;
        float wsnkphy ;
                wsnkphy:long_name = "sinking rate for phytoplankton" ;
                wsnkphy:units = "meter day-1" ;
        float wsnkdet ;
                wsnkdet:long_name = "sinking rate for detritus" ;
                wsnkdet:units = "meter day-1" ;
        float fracrmn ;
                fracrmn:long_name = "fraction of the sinking
phytoplankton and detritus flux that remineralize
s on the seafloor" ;
        float remnnh4 ;
                remnnh4:long_name = "ammonium remineralization
(nitrification) timescale" ;
                remnnh4:units = "day-1" ;
        float remndet ;
                remndet:long_name = "detritus remineralization timescale" ;
                remndet:units = "day-1" ;
        float c2n ;
                c2n:long_name = "Nitrogen:Carbon ratio of phytoplankton" ;
                c2n:units = "moleNitrogen moleCarbon-1" ;
        float c2chl ;
                c2chl:long_name = "Chlorophyll:Carbon ratio" ;
                c2chl:units = "milligramChlorophyll milligramC-1" ;
        int biopos ;
                biopos:long_name = "switch to enforce positive
biological fields" ;
                biopos:option_0 = "do not enforce" ;
                biopos:option_1 = "enforce, once per timestep" ;
                biopos:option_2 = "enforce, during formation of forcing terms" ;
                biopos:option_3 = "kill run if negative fields found" ;
        int iflag(usrdim1) ;
                iflag:long_name = "initialization flags" ;
        int iopt(usrdim1) ;
                iopt:long_name = "tunnable options and switches" ;
        int iout(usrdim2) ;
                iout:long_name = "switches specifing the fields to write out" ;
        int ibiout(usrdim2) ;
                ibiout:long_name = "switches specifing which
biological fields to write out" ;
        int outlev(outlev) ;
                outlev:long_name = "output vertical levels or depths" ;
                outlev:units = "meters, if value is greater than
number of levels" ;
        float zclima(tslev) ;
                zclima:long_name = "depths of mean TS profile" ;
                zclima:units = "meter" ;
        float tclima(tslev) ;
                tclima:long_name = "mean temperature profile" ;
                tclima:units = "Celsius" ;
                tclima:field = "mean temperature, scalar" ;
                tclima:positions = "zclima" ;
        float sclima(tslev) ;
                sclima:long_name = "mean salinity profile" ;
                sclima:units = "PSS" ;
                sclima:field = "mean salinity, scalar" ;
                sclima:positions = "zclima" ;
        float refz(level) ;
                refz:long_name = "depths at center of the flat grid
vertical boxes" ;
                refz:units = "meter" ;
        float hz(level) ;
                hz:long_name = "thicknesses of the flat grid vertical boxes" ;
                hz:units = "meter" ;
        float dxt(tlon) ;
                dxt:long_name = "zonal spacing between tracer points" ;
                dxt:units = "centimeter" ;
        float dyt(tlat) ;
                dyt:long_name = "meridional spacing between tracer points" ;
                dyt:units = "centimeter" ;
        float tbath(tlat, tlon) ;
                tbath:long_name = "bathymetry at tracer points" ;
                tbath:units = "meter" ;
                tbath:field = "bathymetry, scalar" ;
                tbath:positions = "tgrid2" ;
        int landt(tlat, tlon) ;
                landt:long_name = "land/sea mask at tracer points" ;
                landt:options = "0: land, 1: sea" ;
                landt:field = "Tmask, scalar" ;
                landt:positions = "tgrid2" ;
        int landv(vlat, vlon) ;
                landv:long_name = "land/sea mask at velocity points" ;
                landv:options = "0: sea, 1: boundary, 2: land" ;
                landv:field = "Vmask, scalar" ;
                landv:positions = "vgrid2" ;
        float tgrid2(tlat, tlon, axis2) ;
                tgrid2:long_name = "2D grid positions at tracer points" ;
                tgrid2:axis = "1: longitude, 2: latitude" ;
                tgrid2:units = "degrees_east, degrees_north" ;
        float vgrid2(vlat, vlon, axis2) ;
                vgrid2:long_name = "2D grid positions at velocity points" ;
                vgrid2:axis = "1: longitude, 2: latitude" ;
                vgrid2:units = "degrees_east, degrees_north" ;
        float tgrid3(tlat, tlon, outlev, axis3) ;
                tgrid3:long_name = "3D grid positions at tracer points" ;
                tgrid3:axis = "1: longitude, 2: latitude, 3: depth" ;
                tgrid3:units = "degrees_east, degrees_north, meter" ;
        float vgrid3(vlat, vlon, outlev, axis3) ;
                vgrid3:long_name = "3D grid positions at velocity points" ;
                vgrid3:axis = "1: longitude, 2: latitude, 3: depth" ;
                vgrid3:units = "degrees_east, degrees_north, meter" ;
        float wtgrid3(tlat, tlon, outlev, axis3) ;
                wtgrid3:long_name = "3D grid positions at W-tracer points" ;
                wtgrid3:axis = "1: longitude, 2: latitude, 3: depth" ;
                wtgrid3:units = "degrees_east, degrees_north, meter" ;
        float time(time) ;
                time:long_name = "time since initialization" ;
                time:units = "seconds" ;
                time:field = "time, scalar, series" ;
        float pbar(time, tlat, tlon) ;
                pbar:long_name = "transport streamfunction" ;
                pbar:units = "centimeter3 second-1" ;
                pbar:field = "transport, scalar, series" ;
                pbar:positions = "tgrid2" ;
        float vtot(time, vlat, vlon, outlev, vector) ;
                vtot:long_name = "total velocity" ;
                vtot:units = "centimeter second-1" ;
                vtot:field = "total velocity, vector, series" ;
                vtot:positions = "vgrid3" ;
                vtot:_FillValue = 1.e+35f ;
                vtot:missing_value = 1.e+35f ;
        float vclin(time, vlat, vlon, outlev, vector) ;
                vclin:long_name = "baroclinic velocity" ;
                vclin:units = "centimeter second-1" ;
                vclin:field = "baroclinic velocity, vector, series" ;
                vclin:positions = "vgrid3" ;
                vclin:_FillValue = 1.e+35f ;
                vclin:missing_value = 1.e+35f ;
        float wvst(time, tlat, tlon, outlev) ;
                wvst:long_name = "s-coordinate vertical velocity at WT-points" ;
                wvst:units = "centimeter second-1" ;
                wvst:field = "Twvel s-coord, scalar, series" ;
                wvst:positions = "wtgrid3" ;
                wvst:_FillValue = 1.e+35f ;
                wvst:missing_value = 1.e+35f ;
        float wvzt(time, tlat, tlon, outlev) ;
                wvzt:long_name = "vertical velocity at WT-points" ;
                wvzt:units = "centimeter second-1" ;
                wvzt:field = "Twvel, scalar, series" ;
                wvzt:positions = "tgrid3" ;
                wvzt:_FillValue = 1.e+35f ;
                wvzt:missing_value = 1.e+35f ;
        float dena(time, tlat, tlon, outlev) ;
                dena:long_name = "PE density anomaly (sigma-1000)" ;
                dena:units = "kilogram meter-3" ;
                dena:field = "density, scalar, series" ;
                dena:positions = "tgrid3" ;
                dena:missing_value = 1.e+35f ;
        float mld(time, tlat, tlon) ;
                mld:long_name = "mixed-layer depth" ;
                mld:units = "meter" ;
                mld:field = "mixed-layer, scalar, series" ;
                mld:positions = "vgrid2" ;
                mld:_FillValue = 1.e+35f ;
                mld:missing_value = 1.e+35f ;
        float temp(time, tlat, tlon, outlev) ;
                temp:long_name = "temperature" ;
                temp:units = "Celsius" ;
                temp:field = "temperature, scalar, series" ;
                temp:positions = "tgrid3" ;
                temp:_FillValue = 1.e+35f ;
                temp:missing_value = 1.e+35f ;
        float salt(time, tlat, tlon, outlev) ;
                salt:long_name = "salinity" ;
                salt:units = "PSU" ;
                salt:field = "salinity, scalar, series" ;
                salt:positions = "tgrid3" ;
                salt:_FillValue = 1.e+35f ;
                salt:missing_value = 1.e+35f ;
        float NO3(time, tlat, tlon, outlev) ;
                NO3:long_name = "nitrate concentration" ;
                NO3:units = "millimoles nitrogen meter-3" ;
                NO3:field = "nitrate, scalar, series" ;
                NO3:positions = "tgrid3" ;
                NO3:_FillValue = 1.e+35f ;
                NO3:missing_value = 1.e+35f ;
        float CELLNO3(time, tlat, tlon, outlev) ;
                CELLNO3:long_name = "cellular nitrogen (from nitrate)" ;
                CELLNO3:units = "millimoles nitrogen meter-3" ;
                CELLNO3:field = "cellnitrate, scalar, series" ;
                CELLNO3:positions = "tgrid3" ;
                CELLNO3:_FillValue = 1.e+35f ;
                CELLNO3:missing_value = 1.e+35f ;
        float zoo(time, tlat, tlon, outlev) ;
                zoo:long_name = "zooplankton concentration" ;
                zoo:units = "millimoles nitrogen meter-3" ;
                zoo:field = "zooplankton, scalar, series" ;
                zoo:positions = "tgrid3" ;
                zoo:_FillValue = 1.e+35f ;
                zoo:missing_value = 1.e+35f ;
        float NH4(time, tlat, tlon, outlev) ;
                NH4:long_name = "ammonium concentration" ;
                NH4:units = "millimoles nitrogen meter-3" ;
                NH4:field = "ammonium, scalar, series" ;
                NH4:positions = "tgrid3" ;
                NH4:_FillValue = 1.e+35f ;
                NH4:missing_value = 1.e+35f ;
        float detritus(time, tlat, tlon, outlev) ;
                detritus:long_name = "detritus concentration" ;
                detritus:units = "millimoles nitrogen meter-3" ;
                detritus:field = "detritus, scalar, series" ;
                detritus:positions = "tgrid3" ;
                detritus:_FillValue = 1.e+35f ;
                detritus:missing_value = 1.e+35f ;
        float CHL(time, tlat, tlon, outlev) ;
                CHL:long_name = "chlorophyll concentration" ;
                CHL:units = "milligrams meter-3" ;
                CHL:field = "chlorophyll, scalar, series" ;
                CHL:positions = "tgrid3" ;
                CHL:_FillValue = 1.e+35f ;
                CHL:missing_value = 1.e+35f ;
        float CELLNH4(time, tlat, tlon, outlev) ;
                CELLNH4:long_name = "cellular nitrogen (from ammonia)" ;
                CELLNH4:units = "millimoles nitrogen meter-3" ;
                CELLNH4:field = "cellammonia, scalar, series" ;
                CELLNH4:positions = "tgrid3" ;
                CELLNH4:_FillValue = 1.e+35f ;
                CELLNH4:missing_value = 1.e+35f ;
        float NH4pr(time, tlat, tlon, outlev) ;
                NH4pr:long_name = "recycled NH4 production rate" ;
                NH4pr:units = "millimoles nitrogen meter-3 day-1" ;
                NH4pr:field = "NH4 production, scalar, series" ;
                NH4pr:positions = "tgrid3" ;
                NH4pr:_FillValue = 1.e+35f ;
                NH4pr:missing_value = 1.e+35f ;
        float NO3pr(time, tlat, tlon, outlev) ;
                NO3pr:long_name = "new NO3 production rate" ;
                NO3pr:units = "millimoles nitrogen meter-3 day-1" ;
                NO3pr:field = "NO3 production, scalar, series" ;
                NO3pr:positions = "tgrid3" ;
                NO3pr:_FillValue = 1.e+35f ;
                NO3pr:missing_value = 1.e+35f ;
        float zgrphy(time, tlat, tlon, outlev) ;
                zgrphy:long_name = "zooplankton grazing rate of phytoplankton" ;
                zgrphy:units = "millimoles nitrogen meter-3 day-1" ;
                zgrphy:field = "zoo-phyto grazing, scalar, series" ;
                zgrphy:positions = "tgrid3" ;
                zgrphy:_FillValue = 1.e+35f ;
                zgrphy:missing_value = 1.e+35f ;

// global attributes:
                :title = "PE model output fields" ;
                :name = "pe_out.cdf" ;
                :type = "PE MODEL" ;
                :version = 7.27 ;
                :bio_parm = "bioDuse_fct.in" ;
}
'''

SEED_LEVEL = 0

def loadNetCdfData(fvar_str):
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #realizations reader 
    rreader = netcdf_reader.NetcdfReader(pe_dif_sep2_98_file)
    
    #central forecasts reader 
    creader = netcdf_reader.NetcdfReader(pe_fct_aug25_sep2_file)
    fvar8 = creader.readVarArray(fvar_str, 7)
    
    #deviations from central forecast for all 600 realizations
    fvar = rreader.readVarArray(fvar_str)  
    fvar = netcdf_reader.addCentralForecast(fvar, fvar8, level_start=SEED_LEVEL, level_end=SEED_LEVEL) 
    
    return fvar

'''
http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

scipy.stats.spearmanr(a, b=None, axis=0)

Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.

The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two 
datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets 
are normally distributed. Like other correlation coefficients, this one varies between -1 and +1 with 
0 implying no correlation. Correlations of -1 or +1 imply an exact monotonic relationship. Positive 
correlations imply that as x increases, so does y. Negative correlations imply that as x increases, 
y decreases.

The p-value roughly indicates the probability of an uncorrelated system producing datasets that have
a Spearman correlation at least as extreme as the one computed from these datasets. The p-values are
not entirely reliable but are probably reasonable for datasets larger than 500 or so.

Returns:

rho : float or ndarray (2-D square)
Spearman correlation matrix or correlation coefficient (if only 2 variables are given as parameters. 
Correlation matrix is square with length equal to total number of variables (columns or rows) in a and b combined.

p-value : float
The two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated, 
has same dimension as rho.


References

[CRCProbStat2000] Section 14.7

[CRCProbStat2000]    (1, 2) Zwillinger, D. and Kokoska, S. (2000). 
CRC Standard Probability and Statistics Tables and Formulae. Chapman & Hall: New York. 2000.

From http://en.wikipedia.org/wiki/P-value#Definition_and_interpretation:
The smaller the p-value, the larger the significance because it tells 
the investigator that the hypothesis under consideration may not adequately 
explain the observation. The hypothesis H is rejected if any of these probabilities 
is less than or equal to a small, fixed, but arbitrarily pre-defined, threshold value 
\alpha, which is referred to as the level of significance. Unlike the p-value, the 
\alpha level is not derived from any observational data nor does it depend on the 
underlying hypothesis; the value of \alpha is instead determined based on the 
consensus of the research community that the investigator is working in.


'''

from netCDF4 import Dataset as cdfdata

def main():
    
    #realizations reader 
    #rreader = netcdf_reader.NetcdfReader(pe_dif_sep2_98_file)
    #central forecasts reader 
    #creader = netcdf_reader.NetcdfReader(pe_fct_aug25_sep2_file)
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    #pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #float salt(time, tlat, tlon, outlev) ;
    netFile = netCDF4.Dataset(pe_dif_sep2_98_file)

    
    
    '''
    <type 'netCDF4.Variable'>
    float32 temp(time, tlat, tlon, outlev)
    long_name: temperature
    units: Celsius
    field: temperature, scalar, series
    positions: tgrid3
    _FillValue: 1e+35
    missing_value: 1e+35
    unlimited dimensions: time
    current shape = (600, 90, 53, 16)
    filling off
    '''
    
    loc = [ 44, 30 ]
    level =  
    
    tvars = [ 'NO3','temp','salt','zoo','NH4','detritus','CHL', 'CELLNH4', 'CELLNO3']
    vdict = dict()
    
    for v in tvars:
        vdict[v] = np.zeros(shape=600)
        for mem in range(0,60
            vdict[v][mem] = netFile.variables[ v ][ mem ][ loc[0] ][ loc[1] ][ level ]
            
    
    for v1 in tvars:
        for v2 in tvars:
            
            ret = scipy.stats.spearmanr(vdict[v1], vdict[v2], axis=0) 
            pval = ret[1] 
            print 'p-value for {0} and {1} = {2}'.format( v1, v2, pval)

            if pval > 0.05:
                print '\tUNCORRELATED'
            
           
            
            
            
    
    
    
    '''
    temp = np.zeros(shape=600); salt = np.zeros(shape=600)
    uncorr_trial = np.zeros(shape=600)
    for mem in range(0,600):
        temp[mem] = netFile.variables[ 'temp' ][ mem ][ loc[0] ][ loc[1] ][ level ]
        salt[mem] = netFile.variables[ 'salt' ][ mem ][ loc[0] ][ loc[1] ][ level ]
        uncorr_trial[mem] = netFile.variables[ 'NO3' ][ mem ][ loc[0] ][ loc[1] ][ level ]
        
   
    ret1 = scipy.stats.spearmanr(temp, salt, axis=0)   
    ret2 = scipy.stats.spearmanr(temp, uncorr_trial, axis=0)  
    ret3 = scipy.stats.spearmanr(salt, uncorr_trial, axis=0)  
    
    print ret1
    print ret2
    print ret3
    '''
    

    netFile.close()
    print 'finished!'
            
if __name__ == "__main__":  
    main()