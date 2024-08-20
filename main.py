import random
import pandas as pd
from collections import Counter
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import matplotlib.pyplot as plt
from aux import *
from new import *
import os



for i in [32, 33]:
    if not os.path.exists("situation"+str(i)):
        os.makedirs("situation"+str(i))
    situation_scenario(i, "evidence")


#investigate_file("withEvidence/situation9/df.csv")


#create_ground_truth_param_sweep(5)
'''print("param 6")

create_ground_truth_param_sweep(6)
print("param 7")

create_ground_truth_param_sweep(7)
print("param 8")

create_ground_truth_param_sweep(8)'''