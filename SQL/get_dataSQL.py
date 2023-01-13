import mysql.connector
import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
import math

# Connect to the SQLite database

mydb = mysql.connector.connect(
  host="localhost",
  user="sammy2",
  password="Password@123",
  database="xy_lattices"
)

mycursor = mydb.cursor()

# Parameters might change
J = 1
max_t = 1.00
min_t = 2.00
lattice_shape = (8,8) # can be changed to (16,16) or (32,32)
steps = 1
iters_per_step = 30
random_state = 25
t_vals = np.linspace(min_t, max_t, 10)
betas = 1 / t_vals
lattices = []

# Monte Carlo Simulation
for i, beta in enumerate(betas):
    lat=[]
    print(beta)
    random_state=random_state+1
    xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,beta=beta,J=J,random_state=random_state)
    for q in range(30):
        xy.simulate(steps,iters_per_step)
        if q >= 25:
            sql = "INSERT INTO lattices (temperature, lattice) VALUES (%s, %s)"
            val = (t_vals[i], pickle.dumps(xy.L))
            mycursor.execute(sql, val)
    # Insert the lattice and temperature into the database


mydb.commit()
