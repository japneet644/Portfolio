import numpy as np
import math
import gzip
import scipy
from scipy.optimize import least_squares

def get_specific_heat(all_lattices,T):
    #Specific Heat of a set of latticess
    # all_lattices = np.array(all_lattices)
    H_vals=[]
    Lshape = all_lattices[0].shape
    # print(Lshape,T,all_lattices)
    for lattice in all_lattices:
        H_vals.append(0.5*np.sum(calculate_H_matrix(lattice)))
    E_var=np.var(H_vals)
    # print(E_var)
    return (E_var) * (T ** (-2))*(1.0/(Lshape[0]*Lshape[1]))
def modify(lattice):
    mag_x=np.mean(np.cos(2*np.pi*lattice))
    mag_y=np.mean(np.sin(2*np.pi*lattice))
    u = math.atan2(mag_y,mag_x)/(2*np.pi)
    if u<0:
        u = u + 1
    A = lattice - u# + 0.125
    b = A < -0.5
    A[b] = A[b] + 1
    c = A > 0.5
    A[c] = A[c] - 1
    lattice = A
    return lattice

def get_average_correlations(lattice):
    n = 500
    l = lattice.shape[-1]
    lat = []
    for x in lattice:
        lat.append(modify(x))
    lat = np.array(lat)
    lattices = np.stack([np.cos(2*np.pi*lat),np.sin(2*np.pi*lat)],axis=1)
    correlation_matrix = np.zeros((l,l))
    for i in range(l):
        for t in range(l):
            c = 0
            for k in range(n//2):
                c = c + (lattices[k,0,i,t]+ 1j*lattices[k,1,i,t])*(lattices[k+n//2,0,i,t]- 1j*lattices[k+n//2,1,i,t])
            correlation_matrix[i,t] = abs((c/(n//2)))
    return np.mean(correlation_matrix), np.std(correlation_matrix)

def get_energy(all_lattices):
    # print(Lshape,T,all_lattices)
    H_vals=[]
    for lattice in all_lattices:
        H_vals.append(0.5*np.sum(calculate_H_matrix(lattice)))
    return H_vals

def calculate_H_matrix(lattice):
    #H_matrix is the matrix containing the value of hamiltonian at each site
    H_matrix=np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j + 1) % lattice.shape[1]]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j - 1) % lattice.shape[1]]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[(i + 1) % lattice.shape[0], j]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[(i - 1) % lattice.shape[0], j]))
    return H_matrix
def calculate_H1_matrix(lattice):
    #H_matrix is the matrix containing the value of hamiltonian at each site
    H1_matrix=np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            H1_matrix[i, j] -= np.sin(2 * np.pi * (lattice[i, j] - lattice[i, (j + 1) % lattice.shape[1]]))
            H1_matrix[i, j] -= np.sin(2 * np.pi * (lattice[i, j] - lattice[i, (j - 1) % lattice.shape[1]]))
            H1_matrix[i, j] -= np.sin(2 * np.pi * (lattice[i, j] - lattice[(i + 1) % lattice.shape[0], j]))
            H1_matrix[i, j] -= np.sin(2 * np.pi * (lattice[i, j] - lattice[(i - 1) % lattice.shape[0], j]))
    return H1_matrix
def calculate_H2_matrix(lattice):
    #H_matrix is the matrix containing the value of hamiltonian at each site
    H2_matrix=np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            H2_matrix[i, j] += np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j + 1) % lattice.shape[1]]))
            H2_matrix[i, j] += np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j - 1) % lattice.shape[1]]))
            H2_matrix[i, j] += np.cos(2 * np.pi * (lattice[i, j] - lattice[(i + 1) % lattice.shape[0], j]))
            H2_matrix[i, j] += np.cos(2 * np.pi * (lattice[i, j] - lattice[(i - 1) % lattice.shape[0], j]))
    return H2_matrix


def get_magnetization(lattice):
    #Gives net magnetization of a lattice
    mag_x=np.mean(np.cos(2*np.pi*lattice))
    mag_y=np.mean(np.sin(2*np.pi*lattice))
    return (mag_x**2+mag_y**2)**0.5

def get_magnetization_direction(lattices):
    theta=[]
    for lattice in lattices:
        mag_x=np.mean(np.cos(2*np.pi*lattice))
        mag_y=np.mean(np.sin(2*np.pi*lattice))
        u = math.degrees(math.atan2(mag_y,mag_x))
        if u<0:
            theta.append(u+360)
        else:
            theta.append(u)
    return theta

def get_mean_magnetization(lattices):
    #Mean Magnetization and Standard Deviation of a set of lattices
    mag=[]
    for lattice in lattices:
        mag.append(get_magnetization(lattice))
    return [[mag],np.mean(mag),np.std(mag)]#

def get_vorticity_configuration(lattice):
    #Vorticity configuration is the matrix containing the value of vorticity at each site
    vortex_matrix=np.zeros(lattice.shape)
    l=lattice.shape[0]
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,(j+1)%l]-lattice[i,(j+1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,j]-lattice[(i+1)%l,(j+1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,(j-1)%l]-lattice[(i+1)%l,j])
            vortex_matrix[i,j]+=saw(lattice[i,(j-1)%l]-lattice[(i+1)%l,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,(j-1)%l]-lattice[i,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,j]-lattice[(i-1)%l,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,(j+1)%l]-lattice[(i-1)%l,j])
            vortex_matrix[i,j]+=saw(lattice[i,(j+1)%l]-lattice[(i-1)%l,(j+1)%l])
            vortex_matrix[i,j]=round(vortex_matrix[i,j])
    return vortex_matrix

def saw(x):
    #saw function used to calculate vorticity at a site
    if x<=-1/2:
        return x+1
    if x>=1/2:
        return x-1
    else:
        return x
def wasserstein_distance(A,B):
    # sum = 0
    n = len(A)
    dist = np.zeros(n)
    # print(n)
    for x in range(n-1):
        dist[x+1] = A[x]-B[x]+dist[x]
    return np.sum(abs(dist))
def get_correlations(lattices):
    correlations =[]
    for lattice in lattices:
        correlations.append(_compute_space_correlations(lattice))# .reshape((len(4), 1))) ,axis=1)
    # print(correlations)
    mean_corr = np.mean(correlations,axis=0)
    length =get_correlation_length(mean_corr)
    # print(length)
    return (mean_corr,length)
def get_pairwise_correlations(lattices):
    correlations =[]
    n = 30
    mag_x = np.mean(lattices[:,:,:,0],axis = 0)
    mag_y = np.mean(lattices[:,:,:,1],axis = 0)
    for i in range(n):
        for j in range(n):
            correlations.append(0.5*np.mean( (lattices[i,:,:,0] - mag_x)*(lattices[j,:,:,0]-mag_x) + (lattices[i,:,:,1] - mag_y)*(lattices[j,:,:,1]-mag_y)) )
            # correlations.append(compute_correlations(lattices[i],lattices[j]))# .reshape((len(4), 1))) ,axis=1)
    # print(np.mean(correlations,axis=0))
    mean_corr = np.mean(correlations,axis=0)
    # length =get_correlation_length(mean_corr)
    # print(length)
    return mean_corr
def compute_correlations(A,B):
    return np.mean(np.cos(A-B))

def get_helicity_modulus(lattices,T):
    H1_vals=[]
    H2_vals=[]
    Lshape = 8
    # print(Lshape,T,all_lattices)
    for lattice in lattices:
        H1_vals.append(0.5*np.sum(calculate_H1_matrix(lattice)))
        H2_vals.append(0.5*np.sum(calculate_H2_matrix(lattice)))
    E2=np.mean(H2_vals)
    E1_sq=np.mean((np.array(H1_vals))**2)
    return (1.0/(2*8*8))*(E2-E1_sq/T)

def _compute_space_correlations(lattice):
    correlation = []# <(S(x) - <S(x)> ).(S(y) - <S(y)> )>
    # print(lattice)
    lat_cos = np.cos(2*np.pi*lattice)
    lat_sin = np.sin(2*np.pi*lattice)
    mag_x=np.mean(lat_cos)
    mag_y=np.mean(lat_sin)
    # print(mag_x,mag_y)
    rolledh = (lat_cos,lat_sin)
    rolledv = (lat_cos,lat_sin)
    for r in range(5):
        # correlation.append(0.5*np.mean( (lat_cos - mag_x)*(rolledh[0]-mag_x) + (lat_sin - mag_y)*(rolledh[1]-mag_y) +  (lat_cos - mag_x)*(rolledv[0]-mag_x) + (lat_sin - mag_y)*(rolledv[1]-mag_y) ))
        correlation.append(0.5*np.mean( lat_cos*rolledh[0] + lat_sin*rolledh[1] + lat_cos*rolledv[0] + lat_sin*rolledv[1] ))
        rolledh = np.roll(rolledh, 1, axis=2)
        rolledv = np.roll(rolledv, 1, axis=1)
    return correlation
def get_correlation_length(correlation):
    def optimized_func(x, R, f_log):
        return R + x[0] * f_log - x[0] * x[1]
    A = correlation
    bounds = int(len(A) / 5)
    ls = least_squares(optimized_func, [0, 0], kwargs={'R' : np.arange(bounds, len(A)-bounds ),'f_log' : np.log(np.maximum(A[bounds:-bounds], [1e-10] * (len(A) - 2*bounds)))})
    return  ls.x[0]
