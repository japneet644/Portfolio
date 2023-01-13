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
mycursor.execute("SELECT lattice FROM lattices WHERE temperature = 1")
lattice_blob = mycursor.fetchone()[0]
lattice = pickle.loads(lattice_blob)
print(len(lattice))