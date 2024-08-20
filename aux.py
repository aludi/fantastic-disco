import random
import pandas as pd
from collections import Counter
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import pyAgrum.lib.bn2scores as gumscores
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from itertools import chain, combinations, product
import itertools
import copy


def create_ground_truth_param_sweep(ARG):
    unfree_params = ["scn|p&q", "p|scn", "q|scn", "r|scn"]
    params = [0.75, 0.2, 0.01]
    dis = collect_distribution(ARG)
    list_keys = dis.keys()
    free_params = [i for i in list_keys if i not in unfree_params]
    combinations = list(itertools.product(params, repeat=len(free_params)))
    outcomes_list = []
    for combo in combinations:
        for i in range(0,len(free_params)):
            dis[free_params[i]] = combo[i]
        distribution_val = str(dis)
        #distribution = copy.deepcopy(dis)
        full_l = create_distribution(dis, runs=500) # first with 1000 runs but that takes too long (for 1 and 2)
        sig_dif = param_sweep_situation(ARG, full_l)
        if isinstance(sig_dif, pd.DataFrame):
            print(distribution_val)
            #print("found a differnece")
            sig_dif["params"] = distribution_val
            outcomes_list.append(sig_dif)
    if outcomes_list != []:
        outcome_df = pd.concat(outcomes_list)
        outcome_df.to_csv(f"situation{ARG}/paramsweep.csv")
    else:
        print("no difference found in params")



def collect_distribution(ARG):
    if ARG == 1:
        dis = {"p":0.19, "e1|p":0.8, "e1|-p":0.001, "q":0.28, "e2|q":0.8, "e2|-q":0.001, "scn|p&q":1}
    elif ARG == 2:
        dis = {"scn":0.1, "p|scn":1, "q|scn":1, "p|-scn":0.1, "q|-scn":0.2 ,"e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001}
    elif ARG == 3:
        dis = {"p":0.1, "q":0.2, "scn|p&q": 1, "e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001, "e3|p&q":0.9, "e3|p&-q": 0.4}
    elif ARG == 4:
        dis = {"scn":0.3, "p|scn":1, "q|scn":1, "p|-scn":0.1, "q|-scn":0.05, "e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001, "e3|p&q":0.9, "e3|p&-q": 0.4}
    elif ARG == 5:
        dis = {"p":0.1, "q|p":0.7, "q|-p":0.01, "e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001, "scn|p&q":1}
    elif ARG == 6:
        dis = {"scn":0.1, "p|scn":1, "q|scn":1,"p|-scn":0.1, "q|-scn&p":0.7, "q|-scn&-p":0.01,"e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001}
    elif ARG == 7:  # hidden conjunction
        dis = {"scn":0.006, "p|scn": 1, "p|-scn":0.0945, "q|scn":1, "q|-scn":(0.2 - 0.006) / 0.994, "r|scn":1, "e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001}
    elif ARG == 8:  # hidden conjunction with dependency
        dis = {"scn":0.006, "p|scn": 1, "p|-scn":0.4, "q|scn":1, "q|-scn&p":0.4,"q|-scn&-p":0.01,"r|scn":1,"e1|p":0.8, "e1|-p":0.001, "e2|q":0.8, "e2|-q":0.001 }
    elif ARG == 9: # test weird one
        dis = {'p': 0.75, 'q|p': 0.75, 'q|-p': 0.2, 'e1|p': 0.01, 'e1|-p': 0.01, 'e2|q': 0.01, 'e2|-q': 0.01, 'scn|p&q': 1}
    elif ARG == 10:
        dis = {"p":0.3, "q|p&s": 0, "q|p&-s":0.9, "q|-p&s":0.9, "q|-p&-s":0.001, "s":0.4, "r|s&q": 0.8, "t|p&q":0.9, "scn1|p&q&t":1, "scn2|s&q&r":1}
    elif ARG == 11:
        dis = {"p":0.3, "s":0.9999, "scn1|p":0.9, "scn2|s&scn1":0, "scn2|-s&scn1":0, "scn2|s&-scn1":0.7}
    elif ARG == 12 or ARG == 13:
        dis = {"p":0.3, "s|p":0, "s|-p":0.7, "scn1|p":1, "scn2|scn1":0, "scn2|s":1}
    elif ARG == 14: # scn1 : pq, scn2: ts,no dependencies
        dis = {"p":0.3, "q|p":0.8, "q|-p":0.01, "scn1|p&q":1, "scn1|p&-q":0, "scn1|-p&q":0, "scn1|-p&-q":0, "t":0.4, "s|t":0.9, "s|-t":0.1, "scn2|s&t":1, "scn2|s&-t":0,
               "scn2|-s&-t":0,"scn2|-s&t":0}
    elif ARG == 15:# scn1 : pr-q, scn2:qs-p
        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p": 0, "q|-p":0.4, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q": 1, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0}

    elif ARG == 16:  # scn1 : pr-q, scn2:qs-p
        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q":0.4,
               "s|r&q": 0, "s|-r&q": 0.6, "s|-q&-r": 0.1, "s|-q&r": 0, "scn2|s&q": 1, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0}

    elif ARG == 17:  #
        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,

               "s|r&p": 0, "s|-r&p": 0.6, "s|-p&-r": 0.1, "s|-p&r": 0, "scn2|s&p": 1, "scn2|s&-p": 0,
               "scn2|-s&-p": 0, "scn2|-s&p": 0}

        #dis = {"scn1": 0.2, "scn2|scn1":0, "scn2|-scn1":0.4, "scn3|-scn1&-scn2":1,"p|scn1":1, "p|-scn1":0.2, "t|scn2":1, "t|-scn2":0.3}  # de distributie moet ervoor zorgen dat de scenarios uitsluiten

        #dis = {"p": 0.2, "scn1|p":0.8, "scn1|-p":0, "t":0.4, "scn2|t&-scn1":0.7, "scn2|-t&-scn1":0, "scn2|-t&scn1":0, "scn2|t&scn1":0}  # de distributie moet ervoor zorgen dat de scenarios uitsluiten


    elif ARG == 18:  #

        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "k|p&r":0.4, "scn1|p&r&k": 1, "scn1|-k": 0,
               "q|p": 0, "q|-p": 0.6, "t|q":0.6, "t|-q":0.1,
               "s|q&t": 0.9, "scn2|s": 1, "scn2|-s": 0}

    elif ARG == 19:  #

        #dis = {"scn1":0.2, "scn2|scn1":0, "scn2|-scn1":0.6, "p|scn1": 1, "t|scn2":1, "s|scn2":1,
        #       "p|-scn1":0.1, "t|-scn2":0.1, "s|-scn2":0.3, "r|-scn1&s":0,"r|-scn1&-s":0.01, "r|scn1&-s":1}

        dis = {"p":0.3, "r|p":0.4, "r|-p":0.1, "s|p&-r":0.2, "s|-p&-r":0.5, "s|-p&r":0.1, "s|p&r":0,
               "t|s":0.6, "t|-s":0.01, "scn1|p&r":1, "scn2|t&s":1}

        dis = {"p": 0.3, "r|p": 0.4, "r|-p": 0.1, "scn1|p&r":1, "scn1|p&-r":0, "scn1|-p&r":0, "scn1|-p&-r":0,
               "s|scn1":0, "s|-scn1":0.3,
               "t|s": 0.6, "t|-s": 0.01,
               "scn2|t&s": 1, "scn2|-t&-s":0, "scn2|t&-s":0, "scn2|-t&s":0}

    elif ARG == 21:  #

        dis = {"scn1":0.2, "scn2|scn1":0, "scn2|-scn1":0.6, "p|scn1": 1, "t|scn2":1,
              "p|-scn1":0.1, "t|-scn2":0.1,
               "s|scn2&t":1, "s|-t&scn2":0,
               "s|t&-scn2":0, "s|-scn2&-t":0.3,
               "r|-scn1&s&p":0,"r|-scn1&-s&p":0, "r|-scn1&-s&-p":0.01, "r|scn1&-s":1, "r|scn1&s":1,
               "scn1|p&r":1, "scn2|s&t":1}
    elif ARG == 22 or ARG==23 or ARG==24 or ARG == 25:
        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p": 0, "q|-p": 0.4, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q": 1, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0}

    elif ARG == 26:
        dis = {"p": 0.3, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p&-r": 0.5, "q|-p&-r": 0.1, "q|p&r":0, "q|-p&r":0, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q&p": 1, "scn2|s&-q&-p": 0,
               "scn2|-s&-q&-p": 0, "scn2|-s&q&-p": 0,"scn2|s&-q&p": 0,
               "scn2|-s&-q&p": 0, "scn2|-s&q&p": 0}

    elif ARG == 27:
        dis = {"p": 0.4, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 1, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p": 0, "q|-p": 0.45, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q": 1, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0, "e1|p":0.9, "e1|-p":0.01, "e2|r":0.9, "e2|-r":0.02,
               "e3|q":0.9, "e3|-q":0.01, "e4|s":0.9, "e4|-s":0.01}

    elif ARG == 28:
        dis = {"p": 0.4, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 0.9, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p": 0, "q|-p": 0.45, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q": 0.9, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0, "e1|p": 0.9, "e1|-p": 0.01, "e2|r": 0.9, "e2|-r": 0.02,
               "e3|q": 0.9, "e3|-q": 0.01, "e4|s": 0.9, "e4|-s": 0.01}

    elif ARG == 29:
        dis = {"p": 0.4, "r|p": 0.8, "r|-p": 0.01, "scn1|p&r": 0.9, "scn1|p&-r": 0, "scn1|-p&r": 0, "scn1|-p&-r": 0,
               "q|p": 0, "q|-p": 0.45, "s|q": 0.9, "s|-q": 0.1, "scn2|s&q": 0.9, "scn2|s&-q": 0,
               "scn2|-s&-q": 0, "scn2|-s&q": 0}

    elif ARG == 30:  #

        dis = {"scn1":0.3, "scn2|scn1":0, "scn2|-scn1":0.6,
               "p|scn1": 1, "r|scn1":1, "t|scn2&-p":1, "t|scn2&p":0, "s|scn2":1,
               "p|-scn1":0.1,
               "r|-scn1&p": 0, "r|-scn1&-p": 0.01,
               "t|-scn2&-p":0.1, "t|-scn2&p":0,
               "s|t&-scn2":0, "s|-scn2&-t":0.3,
               "e1|p":0.9, "e1|-p":0.01, "e2|r":0.9, "e2|-r":0.02,
               "e3|t":0.9, "e3|-t":0.01, "e4|s":0.9, "e4|-s":0.01}

    elif ARG == 31:  #

        dis = {"scn1":0.3, "scn2|scn1":0, "scn2|-scn1":0.6,
               "p|scn1": 1, "r|scn1":1,  "s|scn2":1,
               "p|-scn1":0.1,
               "t|scn2&-p": 1, "t|scn2&p": 0,
               "r|-scn1&p": 0, "r|-scn1&-p": 0.01,
               "t|-scn2&-p":0.1, "t|-scn2&p":0,
               "s|t&-scn2":0, "s|-scn2&-t":0.3}


    elif ARG == 39:  #

        dis = {"scn1":0.3, "scn2|scn1":0, "scn2|-scn1":0.6,
               "p|scn1&-scn2": 1, "p|scn1&scn2": 0,
               "r|scn1": 1, "s|scn2": 1,
               "p|-scn1&-scn2": 0.1, "p|-scn1&scn2": 0,
               "q|scn2&-p": 1, "q|scn2&p": 0,
               "q|-scn2&-p": 0.1, "q|-scn2&p": 0,
               "r|-scn1&p": 0, "r|-scn1&-p": 0.01,
               "s|q&-scn2":0, "s|-scn2&-q":0.3}

    elif ARG==32 or ARG == 33:  #

        dis = {"scn1":0.3, "scn2|scn1":0, "scn2|-scn1":0.6,
               "p|scn1&-scn2": 1, "p|scn1&scn2": 0,
               "r|scn1":1,  "s|scn2":1,
               "p|-scn1&-scn2":0.1,"p|-scn1&scn2":0,
               "q|scn2&-p": 1, "q|scn2&p": 0,
               "q|-scn2&-p": 0.1, "q|-scn2&p": 0,
               "r|-scn1&p": 0, "r|-scn1&-p": 0.01,
               "s|q&-scn2":0, "s|-scn2&-q":0.3,
               "e1|p": 0.9, "e1|-p": 0.01, "e2|r": 0.9, "e2|-r": 0.01,
               "e3|q": 0.9, "e3|-q": 0.01, "e4|s": 0.9, "e4|-s": 0.01
               }

    elif ARG == 34: # independent scenairos

        dis = {"scn1": 0.3, "scn2": 0.6,
              "p|scn1":1, "p|-scn1":0.3,
               "q|scn1":1, "q|-scn1&p":0.6, "q|-scn1&-p":0.1,
               "t|scn2":1, "t|-scn2":0.3,
               "s|scn2":1, "s|-scn2&t":0.7, "s|-scn2&-t":0.1
               }


    else:
        return 0
    return dis


def arc_structure(sit_arg, df, ev_at):
    if sit_arg in [1, 2, 4]:
        arcs_scn = [("p", "e1"), ("q", "e2"), ("scn", "p"), ("scn", "q")]
        arcs_alt = [("p", "e1"), ("q", "e2"), ("p", "scn"), ("q", "scn")]
        scn_df = df
        alt_df = df
    elif sit_arg in [5, 6, 9]:
        arcs_scn = [("p", "e1"), ("q", "e2"), ("scn", "p"), ("scn", "q"), ("p", "q")]
        arcs_alt = [("p", "e1"), ("q", "e2"), ("p", "scn"), ("q", "scn"), ("p", "q")]
        scn_df = df
        alt_df = df
    elif sit_arg in [7]:
        arcs_scn = [("p", "e1"), ("q", "e2"), ("scn", "p"), ("scn", "q"), ("p","q")]
        scn_df = df.drop(["r"], axis=1)
        arcs_alt = [("p", "e1"), ("q", "e2"), ("p", "r"), ("q", "r"), ("p", "q"), ("r", "scn")]
        alt_df = df
    elif sit_arg in [8]:
        arcs_scn = [("p", "e1"), ("q", "e2"), ("scn", "p"), ("scn", "q"), ("p", "q")]
        scn_df = df.drop(["r"], axis=1)
        arcs_alt = [("p", "e1"), ("q", "e2"), ("p", "r"), ("q", "r"), ("r", "scn"), ("p", "q")]
        alt_df = df
    elif sit_arg in [10]:
        arcs_scn = [("p", "q"), ("s", "q"), ("scn1", "p"),
                     ("scn1", "q"), ("scn2", "q"), ("scn2", "s")]
        scn_df = df.drop(["r", "t"], axis=1)
        arcs_alt = [("p", "q"), ("s", "q"), #("p", "s"),
                    ("p", "scn1"), ("q", "scn1"), ("s","scn2"), ("q","scn2"), ("scn1", "scn2")
                    ]
        alt_df = df.drop(["r", "t"], axis=1)
    elif sit_arg in [11]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "s")]
        scn_df = df
        arcs_alt = [
                    ("p", "scn1"), ("s","scn2"), ("scn1","scn2")]
        alt_df = df
    elif sit_arg in [12]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "s")]
        scn_df = df
        arcs_alt = [
                    ("p", "scn1"), ("s","scn2"), ("scn1","scn2")]
        alt_df = df
    elif sit_arg in [13]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "s"), ("p", "s")]
        scn_df = df
        arcs_alt = [
                    ("p", "scn1"), ("s","scn2"), ("p","s")]
        alt_df = df
    elif sit_arg in [14]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "t")]
        scn_df = df.drop(["q", "s"], axis=1)
        arcs_alt = [
                    ("q","scn1"),("p", "q"), ("p","scn1"), ("s","scn2"),("t","scn2"),("t","s")]
        alt_df = df

    elif sit_arg in [15]:
        arcs_scn = [ ("scn1", "p"), ("scn2", "p"), ("scn1", "q"),
                      ("scn2", "q"), ("p", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p", "r"), ("p", "q"), ("q","scn2"), ("s","scn2"), ("q", "s")]
        alt_df = df

    elif sit_arg in [16]:
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p","r"),
                    ("scn2", "q"), ("scn2","s"), ("q","s"), ("r","s")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        # scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p", "r"), ("r", "s"), ("q", "scn2"), ("s", "scn2"), ("q", "s")]

        # test if its just the constraint node thats messing shit up by replacing alt with scn


        alt_df = df

    elif sit_arg in [17]:
        arcs_scn = [("scn1", "p"),
                    ("scn2", "p")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df.drop(["r", "s"], axis=1)
        # scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p", "r"), ("p", "s"), ("p", "scn2"), ("s", "scn2"), ("r", "s")]
        alt_df = df

    elif sit_arg in [18]:
        arcs_scn = [("scn1", "p"), ("scn1", "r"),
                    ("scn2", "q"), ("scn2", "t")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df.drop(["k", "s"], axis=1)
        # scn_df = df
        arcs_alt = [("p", "k"), ("r", "k"), ("p","r"), ("k","scn1"),
                    ("p", "q"), ("q", "t"), ("t", "s"), ("q","s"), ("s", "scn2")]
        alt_df = df

    elif sit_arg in [19]:
        arcs_scn = [("scn1", "p"), ("scn1", "r"),
                    ("scn2", "s"), ("scn2", "t")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df
        arcs_alt = [("p", "r"), ("r","scn1"), ("p","scn1"),
                    ("t", "s"), ("t","scn2"), ("s", "scn2"),
                    ("p", "s"), ("p","t"), ("r","s"), ("r","t")]

        arcs_alt = [("p", "r"), ("r", "scn1"), ("p", "scn1"),
                    ("t", "s"), ("t", "scn2"), ("s", "scn2"),
                    ("p", "s"), ("r","s"), ("p", "t"), ("r", "t")]

        alt_df = df


    elif sit_arg in [21]:
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "t"), ("s", "t")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df


        arcs_alt = [("p", "r"), ("r", "scn1"), ("p", "scn1"),
                    ("t", "s"), ("t", "scn2"), ("s", "scn2"),
                    ("p", "s"), ("r","s"), ("p", "t"), ("r", "t"), ("scn1", "scn2")]

        arcs_alt = [("p", "r"), ("r", "scn1"), ("p", "scn1"),
                    ("t", "s"), ("t", "scn2"), ("s", "scn2"),
                    ("s", "r"), ("p", "s"), ("p", "t"), ("t", "r")]
        alt_df = df

    elif sit_arg in [22]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "q"), ("p", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p", "r"), ("p", "q"), ("q","scn2"), ("s","scn2"), ("q", "s")]
        alt_df = df


    elif sit_arg in [23]:
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p", "r"), ("p", "q"), ("q","scn2"), ("s","scn2"), ("q", "s")]
        alt_df = df

    elif sit_arg in [24]:   # alt = scenario without scenario node
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("scn1", "p"), ("scn1", "scn2"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")]
        alt_df = df

    elif sit_arg in [25]:   # alt = scenario without scenario node with p->q
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("scn1", "p"), ("scn1", "scn2"), ("p","q"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")]
        alt_df = df


    elif sit_arg in [26]:   # same event p in both scenarios
        arcs_scn = [ ("scn1", "p"), ("scn2", "p"), ("p","q"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s")] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("p", "scn1"), ("r", "scn1"), ("p","r"), ("p","scn2"), ("q","scn2"), ("p","q"),
                     ("s","scn2"), ("q","s"), ("r","q")]
        alt_df = df


    elif sit_arg in [27] or sit_arg in [28]:   # test event final
        arcs_scn = [("scn1", "p"),("scn1", "r"), ("p", "r"),
                    ("scn2", "q"), ("scn2", "s"),("q", "s"),
                    ("p","e1"), ("r","e2"), ("q","e3"), ("s","e4")]
        scn_df = df  #
        # scn_df = df
        arcs_alt = [("p", "scn1"),("r", "scn1"), ("p", "r"), ("p","q"),
                    ("q", "scn2"), ("s", "scn2"), ("q", "s"),
                    ("p","e1"), ("r","e2"), ("q","e3"), ("s","e4")]
        alt_df = df


    elif sit_arg in [29]:   # alt = scenario without scenario node with p->q
        arcs_scn = [ ("scn1", "p"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s"),

                     ] # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["r", "s"], axis=1)
        #scn_df = df
        arcs_alt = [("scn1", "p"), ("scn1", "scn2"), ("p","q"),
                      ("scn2", "q"), ("scn1", "r"), ("p","r"), ("scn2","s"), ("q","s"),
                    ]
        alt_df = df

    elif sit_arg in [30]:
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "t"), ("s", "t"), ("p","e1"), ("r","e2"), ("t","e3"), ("s","e4")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df


        arcs_alt = [("p", "r"), ("r", "scn1"), ("p", "scn1"),
                    ("t", "s"), ("t", "scn2"), ("s", "scn2"),
                    ("p","s"), ("r","t"), ("r","s"),
                    ("p", "t"), ("p","e1"), ("r","e2"), ("t","e3"), ("s","e4")]
        alt_df = df


    elif sit_arg in [31]: # ALT is SCENARIO
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "t"), ("s", "t")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df


        arcs_alt = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "t"), ("s", "t"), ("p","t")]
        alt_df = df


    elif sit_arg in [39]: # ALT is SCENARIO
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "q"), ("q", "s")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df

        arcs_alt = [("scn1", "p"), ("scn1", "r"), ("p", "r"), ("scn1", "scn2"),
                    ("scn2", "s"), ("scn2", "q"), ("q", "s"), ("p","q")]
        alt_df = df

    elif sit_arg in [32]: # ALT is SCENARIO
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "q"), ("q", "s"),
                    ("p","e1"), ("r","e2"), ("q","e3"), ("s","e4")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df

        arcs_alt = [("scn1", "p"), ("scn1", "r"), ("p", "r"), ("scn1", "scn2"),
                    ("scn2", "s"), ("scn2", "q"), ("q", "s"), ("p","q"),
                    ("p","e1"), ("r","e2"), ("q","e3"), ("s","e4")]
        alt_df = df

    elif sit_arg in [33]: # final/current bn
        arcs_scn = [("scn1", "p"), ("scn1", "r"), ("p", "r"),
                    ("scn2", "s"), ("scn2", "q"), ("q", "s"),
                    ("p","e1"), ("r","e2"), ("q","e3"), ("s","e4")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df #df.drop(["s", "r"], axis=1)
        # scn_df = df


        arcs_alt = [("p", "r"), ("p", "scn1"), ("r", "scn1"), ("p", "q"),
                    ("p", "s"), ("r","s"),
                    ("s", "scn2"), ("q", "scn2"), ("q", "s"),
                    ("p", "e1"), ("r", "e2"), ("q", "e3"), ("s", "e4")
                    ]
        alt_df = df

    elif sit_arg in [34]:

        arcs_scn = [("scn1", "p"), ("scn1", "q"), ("p", "q"),
                    ("scn2", "s"), ("scn2", "t"), ("t", "q")]  # add ("scn1", "q"), ("scn2", "p") -> didn't work
        scn_df = df  # df.drop(["s", "r"], axis=1)
        # scn_df = df

        arcs_alt = [("p", "q"), ("p", "scn1"), ("q", "scn1"),
                    ("s", "scn2"), ("t", "scn2"), ("t", "s")
                    ]
        alt_df = df

        '''"scn1": 0.3, "scn2": 0.6,
              "p|scn1":1, "p|-scn1":0.3,
               "q|scn1":1, "q|-scn1&p":0.6, "q|-scn1&-p":0.1,
               "t|scn2":1, "t|-scn2":0.3,
               "s|scn2":1, "s|-scn2&t":0.7, "s|-scn2&-t":0.1'''



    elif sit_arg in [20]:
        arcs_scn = [("p", "q"), ("q", "r"), ("p", "r"), ("s", "r"), ("t", "s"), ("u","s"), ("scn1", "p"),
        ("scn1","q"), ("scn1", "r"), ("scn2", "r"), ("scn2", "s"), ("scn2","t"), ("scn2", "u"), ("scn1", "con"), ("scn2", "con")]
        scn_df = df
        arcs_alt = [("p", "q"), ("q", "r"), ("p", "r"), ("s", "r"), ("t", "s"), ("u","s")]
        alt_df = df.drop(["scn1", "scn2", "con"], axis=1)



    return arcs_scn, arcs_alt, scn_df, alt_df


def investigate_file(file_name):
    df = pd.read_csv(file_name)
    evidence_on = df[(df["e1"] == 1) & (df["e2"] == 1)]
    print(evidence_on)
    print(evidence_on.value_counts())

def create_ground_truth_distribution(ARG, ev_at):
    dis = collect_distribution(ARG)
    if ev_at != "evidence": # removing evidence items from the distribution
        l = []
        for k in dis.keys():
            if "e" in k:
                l.append(k)
        for k in l:
            dis.pop(k)
    return create_distribution(dis)



def create_distribution(dis, runs=10000):

    # the ordering should always be that the conditioning factors should have the chance to go first

    atom_dict = {}
    cond_dict = {}
    for x in dis.keys():
        if "|" not in x:
            atom_dict[x] = dis[x]
        else:
            cond_dict[x] = dis[x]

    full_l = []
    runs = 1000000 # normal : 100000
    for i in range(0, runs):    # generate distribution
        l = []
        for atom in atom_dict:
            if random.random() <= atom_dict[atom]:
                l.append(atom)
            else:
                l.append("-"+atom)

        for c in cond_dict:
            [atom, cond] = c.split("|")
            aslistcond = cond.split("&")
            if all(e in l for e in aslistcond): # if the condition matches
                if random.random() <= cond_dict[c]:
                    if atom not in l:
                        l.append(atom)
                else:
                    if "-"+atom not in l:
                        l.append("-"+atom)
        full_l.append(l)
    print(full_l)
    #exit()
    return full_l


def calculate_frequencies_sprim(df, post_var, relevant_evidence):   # post_var = scn
    outcome_dict = {}

    ev_list = relevant_evidence

    for ev_dict in ev_list:
        query = ""
        for k in ev_dict.keys():
            v = ev_dict[k]
            query = query + k + " ==" + str(v) + " and "
        query = query + "p == p"

        dfp = df.query(query)
        r = dfp[post_var].value_counts() / dfp[post_var].value_counts().sum()

        try:
            post = r[1]
        except KeyError:
            print("posterior frequency of ", ev_dict, " is 0")
            post = 0
        else:
            post = r[1]
        outcome_dict[str(ev_dict)] = post

    return outcome_dict


def calculate_posterior_sprim(bn, post_node, relevant_evidence):
    outcome_dict = {}
    ev_prob_dict = {}

    ev_list = relevant_evidence

    for ev_dict in ev_list:
        ie = gum.LazyPropagation(bn)
        # print("prior ",post_node,  ie.posterior(post_node)[1], "given", ev_dict)

        try:
            ie.updateEvidence(ev_dict)
        except Exception as e:
            print(e)
            print("go back with prob of 0")
            outcome_dict[str(ev_dict)] = 0
            #return outcome_dict

        if 'constraint' in list(bn.names()):
            ie.addEvidence("constraint", [1, 1, 0])

            # print("posterior" ,post_node, ie.posterior(post_node)[1], "given", ev_dict)

            try:
                gumimage.exportInference(bn, f"out/scnConstr{post_node}{ev_dict}.png", evs = {**ev_dict, **{'constraint':[1,1,0]}})
            except Exception:
                print("incompatible evidence")
        else:
            try:
                gumimage.exportInference(bn, f"out/scnConj{post_node}{ev_dict}.png", evs = ev_dict)

            except Exception:
                print("incompatible evidence")

        #print(ev_dict, post_node, ie.posterior("scn1"))
        #print(ev_dict, post_node, ie.posterior("scn2"))


        try:
            print(ev_dict, post_node, ie.posterior(post_node))
            print("evidence probability", ie.evidenceProbability())
            ev_prob_dict[str(ev_dict)] = ie.evidenceProbability()
            outcome_dict[str(ev_dict)] = ie.posterior(post_node)[1]
            if ie.evidenceProbability() <= 0.0:
                outcome_dict[str(ev_dict)] = 0
        except Exception:
            outcome_dict[str(ev_dict)] = 0

    return outcome_dict, ev_prob_dict


def calculate_frequencies_alternative_scenarios():
    pass




def calculate_posterior(bn, post_nodes):
    outcome_dict = {}

    ev_list = [{}, {"e1": 0}, {"e1": 1}, {"e2": 0}, {"e2": 1}, {"e3": 0}, {"e3": 1},
               {"e1": 0, "e2": 0}, {"e1": 0, "e2": 1}, {"e1": 1, "e2": 0}, {"e1": 1, "e2": 1},
               {"e1": 0, "e3": 0}, {"e1": 0, "e3": 1}, {"e1": 1, "e3": 0}, {"e1": 1, "e3": 1},
               {"e3": 0, "e2": 0}, {"e3": 0, "e2": 1}, {"e3": 1, "e2": 0}, {"e3": 1, "e2": 1},
               {"e1": 0, "e2": 0, "e3": 0}, {"e1": 0, "e2": 0, "e3": 1}, {"e1": 0, "e2": 1, "e3": 0},
               {"e1": 0, "e2": 1, "e3": 1},
               {"e1": 1, "e2": 0, "e3": 0}, {"e1": 1, "e2": 0, "e3": 1}, {"e1": 1, "e2": 1, "e3": 0},
               {"e1": 1, "e2": 1, "e3": 1}]

    for ev_dict in ev_list:
        ie = gum.LazyPropagation(bn)
        # print("prior ",post_node,  ie.posterior(post_node)[1], "given", ev_dict)

        ie.updateEvidence(ev_dict)

        # print("posterior" ,post_node, ie.posterior(post_node)[1], "given", ev_dict)

        outcome_dict[str(ev_dict)] = ie.jointPosterior(post_nodes)[1]

    return outcome_dict


def calculate_frequencies(df):
    outcome_dict = {}

    ev_list = [{}, {"e1": 0}, {"e1": 1}, {"e2": 0}, {"e2": 1}, {"e3": 0}, {"e3": 1},
               {"e1": 0, "e2": 0}, {"e1": 0, "e2": 1}, {"e1": 1, "e2": 0}, {"e1": 1, "e2": 1},
               {"e1": 0, "e3": 0}, {"e1": 0, "e3": 1}, {"e1": 1, "e3": 0}, {"e1": 1, "e3": 1},
               {"e3": 0, "e2": 0}, {"e3": 0, "e2": 1}, {"e3": 1, "e2": 0}, {"e3": 1, "e2": 1},
               {"e1": 0, "e2": 0, "e3": 0}, {"e1": 0, "e2": 0, "e3": 1}, {"e1": 0, "e2": 1, "e3": 0},
               {"e1": 0, "e2": 1, "e3": 1},
               {"e1": 1, "e2": 0, "e3": 0}, {"e1": 1, "e2": 0, "e3": 1}, {"e1": 1, "e2": 1, "e3": 0},
               {"e1": 1, "e2": 1, "e3": 1}]

    for ev_dict in ev_list:
        query = ""
        for k in ev_dict.keys():
            v = ev_dict[k]
            query = query + k + " ==" + str(v) + " and "
        query = query + "p == p"

        dfp = df.query(query)
        r = dfp["pq"].value_counts() / dfp["pq"].value_counts().sum()

        try:
            post = r[1]
        except KeyError:
            print("key error for ", ev_dict)
            post = 0
        else:
            post = r[1]
        outcome_dict[str(ev_dict)] = post

    return outcome_dict

    # print(outcome_dict)

def learn_bn(df, arcs, name):
    bn = gum.BayesNet()
    b = gum.BNLearner(df)

    for arcs in arcs:
        (head_a, tail_a) = arcs
        if head_a not in bn.names():
            bn.add(head_a)
        if tail_a not in bn.names():
            bn.add(tail_a)
        bn.addArc(head_a, tail_a)

    b.useSmoothingPrior(0.000000000000001)  # this is vaguely annoying #0.000000000001
    print(bn.names())
    print(df.columns)
    bn = b.learnParameters(bn)

    # get the 0's into the scn
    for node in bn.names():
        for i in bn.cpt(node).loopIn():
            nv = bn.cpt(node).get(i)
            if nv < 0.0000000000001:
                bn.cpt(node).set(i, 0)

    if name != "0":
        gum.saveBN(bn, f"{name}.net")
        gumimage.exportInference(bn, f"{name}.png")
    return bn


def venn_diagrams(sit_arg, df):
    folder=f"situation{sit_arg}"
    l = df.shape[0]
    p = df[(df["p"] == 1) & (df["q"] == 0) & (df["scn"] == 0)]
    q = df[(df["q"] == 1)& (df["p"] == 0) & (df["scn"] == 0)]
    scn = df[(df["scn"] == 1)& (df["q"] == 0) & (df["p"] == 0)]
    pq = df[(df["p"]==1) & (df["q"] == 1) & (df["scn"] == 0)]
    pscn = df[(df["p"]==1) & (df["scn"] == 1)& (df["q"] == 0)]
    qscn = df[(df["q"]==1) & (df["scn"] == 1)& (df["p"] == 0)]
    all = df[(df["p"]==1) & (df["q"]==1) & (df["scn"] == 1)]

    f = open(f"{folder}/marginals.txt", 'w')

    s = f'marginal probabilities from data \n' \
        f'F(p), {df["p"].value_counts()/l} \n' \
        f'F(q), {df["q"].value_counts()/l} \n' \
        f'F(scn), {df["scn"].value_counts()/l}'

    print(s, file=f)
    #plt.figure()
    venn3(subsets=(len(p)/l, len(q)/l, len(pq)/l, len(scn)/l, len(pscn)/l, len(qscn)/l, len(all)/l), set_labels=("p", "q", "scn"), alpha=0.5)
    plt.savefig(f"{folder}/venn.png")

    #plt.show()




def convert_networks_to_hugin(sit_arg):
    for nw in ["alt", "scn"]:
        file_path = f"situation{sit_arg}/{nw}.net"
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Remove the first three lines
        lines[2] = f"name = \"{sit_arg}{nw}\";\n"

        for i, line in enumerate(lines):
            if "states" in line:
                if "(0 1 )" in line:
                    lines[i] = line.replace("(0 1 )", "(\"0\" \"1\")")
                else:
                    lines[i] = line.replace("(0 1 2 )", "(\"0\" \"1\" \"2\")")

        with open(f"situation{sit_arg}/hugin{nw}.net", 'w') as file:
            file.writelines(lines)

def truth_values(subset):
    if not subset:
        return [{}]
    return [{prop: val for prop, val in zip(subset, assignment)}
            for assignment in product([0, 1], repeat=len(subset))]

def get_relevant_evidence(names):
    names = [e for e in names if e not in ['scn1', 'scn2']] # remove setting evidence on scenario nodes
    if "e1" in names:
        names = [e for e in names if e in ['e1', "e2", "e3", "e4"]] # for arg 27 only set on evidence

    p_set = list(chain.from_iterable(combinations(names,r) for r in range(len(names)+1)))
    combi = []
    for subset in p_set:
        combi.extend(truth_values(subset))
    return combi

def situation_scenario(sit_arg, evidence_attached):

    folder = f"situation{sit_arg}"

    full_l = create_ground_truth_distribution(sit_arg, evidence_attached)

    df = pd.DataFrame(map(Counter, full_l)).fillna(0, downcast='infer')
    cols = [c for c in df.columns if c[0] != "-"]
    df = df[cols]
    df.to_csv(f"{folder}/df.csv", index=False)
    # visualize using venn diagrams

    # relevant evidence

    relevant_evidence = get_relevant_evidence(df.columns)
    print(relevant_evidence)

    if sit_arg > 10:    # more than 1 scenario

        #### calculate frequencies:
        out_f1 = calculate_frequencies_sprim(df, 'scn1',relevant_evidence)
        out_f2 = calculate_frequencies_sprim(df, 'scn2',relevant_evidence)

        outf1 = pd.DataFrame.from_dict(out_f1, orient='index')
        outf2 = pd.DataFrame.from_dict(out_f2, orient='index')
        outf = pd.concat([outf1, outf2], axis=1)
        outf.to_csv(f"{folder}/freq")

        #### BN learner #######
        arcs_scn, arcs_alt, scn_df, alt_df = arc_structure(sit_arg, df, evidence_attached)

        # add constraint node
        print(scn_df.head())
        bn_scn = learn_bn(scn_df, arcs_scn, f"{folder}/scn")
        print(alt_df.head())
        bn_alt = learn_bn(alt_df, arcs_alt, f"{folder}/alt")

        # add constraint node to scn
        bn_scn.add(gum.LabelizedVariable('constraint', 'constraint', 3))
        bn_scn.addArc('scn1', 'constraint')
        bn_scn.addArc('scn2', 'constraint')
        bn_scn.cpt("constraint")[{'scn1':0, "scn2":0}] = [0.5, 0.5, 0]
        bn_scn.cpt("constraint")[{'scn1':1, "scn2":0}] = [1, 0, 0]
        bn_scn.cpt("constraint")[{'scn1':0, "scn2":1}] = [0, 1, 0]
        bn_scn.cpt("constraint")[{'scn1':1, "scn2":1}] = [0, 0, 1]
        gum.saveBN(bn_scn, f"{folder}/scn.net")
        gumimage.exportInference(bn_scn, f"{folder}/scn.png", evs={'constraint':[1,1,0]})


        convert_networks_to_hugin(sit_arg)


        # print scores
        '''f = open(f"{folder}/scores_scn.txt", 'w')
        print(gumscores.computeScores(bn_scn, f"{folder}/df.csv")[1], file=f)
    
        f = open(f"{folder}/scores_alt.txt", 'w')
        print(gumscores.computeScores(bn_alt, f"{folder}/df.csv")[1], file=f)'''

        ####### inference with evidence # scn interpretation = conj


        out_scn1, ev_prob_dict_scn = calculate_posterior_sprim(bn_scn, 'scn1', relevant_evidence)
        out_scn2, ev_prob_dict_scn = calculate_posterior_sprim(bn_scn, 'scn2', relevant_evidence)

        out_alt1, ev_prob_dict_conj = calculate_posterior_sprim(bn_alt, 'scn1', relevant_evidence)
        out_alt2, ev_prob_dict_conj = calculate_posterior_sprim(bn_alt, 'scn2', relevant_evidence)

        outdfscn1 = pd.DataFrame.from_dict(out_scn1, orient='index')
        outdfscn2 = pd.DataFrame.from_dict(out_scn2, orient='index')
        outdfevprobscn = pd.DataFrame.from_dict(ev_prob_dict_scn, orient='index')

        outscn = pd.concat([outdfscn1, outdfscn2, outdfevprobscn], axis=1)

        outdfalt1 = pd.DataFrame.from_dict(out_alt1, orient='index')
        outdfalt2 = pd.DataFrame.from_dict(out_alt2, orient='index')
        outdfevprobconj = pd.DataFrame.from_dict(ev_prob_dict_conj, orient='index')
        outalt = pd.concat([outdfalt1, outdfalt2, outdfevprobconj], axis=1)

        plot_scenario_outcomes([outf, outscn, outalt], f"{folder}/")
        plot_outcomes_from_csv(f"{folder}/")
    else:
        #### calculate frequencies:
        out_f1 = calculate_frequencies_sprim(df, 'scn', relevant_evidence)

        outf1 = pd.DataFrame.from_dict(out_f1, orient='index')
        outf = pd.concat([outf1], axis=1)
        outf.to_csv(f"{folder}/freq")

        #### BN learner #######
        arcs_scn, arcs_alt, scn_df, alt_df = arc_structure(sit_arg, df, evidence_attached)

        # add constraint node
        print(scn_df.head())
        bn_scn = learn_bn(scn_df, arcs_scn, f"{folder}/scn")
        print(alt_df.head())
        bn_alt = learn_bn(alt_df, arcs_alt, f"{folder}/alt")


        convert_networks_to_hugin(sit_arg)

        # print scores
        '''f = open(f"{folder}/scores_scn.txt", 'w')
        print(gumscores.computeScores(bn_scn, f"{folder}/df.csv")[1], file=f)

        f = open(f"{folder}/scores_alt.txt", 'w')
        print(gumscores.computeScores(bn_alt, f"{folder}/df.csv")[1], file=f)'''

        ####### inference with evidence # scn interpretation = conj

        out_scn1, ev_prob_scn = calculate_posterior_sprim(bn_scn, 'scn', relevant_evidence)

        out_alt1, ev_prob_conj = calculate_posterior_sprim(bn_alt, 'scn', relevant_evidence)

        outdfscn1 = pd.DataFrame.from_dict(out_scn1, orient='index')
        outdfevscn = pd.DataFrame.from_dict(ev_prob_scn, orient='index')

        outscn = pd.concat([outdfscn1, outdfevscn], axis=1)

        outdfalt1 = pd.DataFrame.from_dict(out_alt1, orient='index')
        outdfevconj = pd.DataFrame.from_dict(ev_prob_conj, orient='index')

        outalt = pd.concat([outdfalt1, outdfevconj], axis=1)

        plot_scenario_outcomes([outf, outscn, outalt], f"{folder}/")
        #plot_outcomes_from_csv(f"{folder}/")

def situation(sit_arg, evidence_attached):
    print("plot individual outcomes of situation")
    folder = f"situation{sit_arg}"

    if evidence_attached == "evidence":
        relevant_evidence = [{}, {"e1": 0}, {"e1": 1}, {"e2": 0}, {"e2": 1},
                         {"e1": 0, "e2": 0}, {"e1": 0, "e2": 1}, {"e1": 1, "e2": 0}, {"e1": 1, "e2": 1}]
    else:
        relevant_evidence = [{}, {"p": 0}, {"p": 1}, {"q": 0}, {"q": 1},
                         {"p": 0, "q": 0}, {"p": 0, "q": 1}, {"p": 1, "q": 0}, {"p": 1, "q": 1}]


    full_l = create_ground_truth_distribution(sit_arg, evidence_attached)

    df = pd.DataFrame(map(Counter, full_l)).fillna(0, downcast='infer')
    cols = [c for c in df.columns if c[0] != "-"]
    df = df[cols]
    df.to_csv(f"{folder}/df.csv", index=False)
    # visualize using venn diagrams
    venn_diagrams(sit_arg, df)


    #### calculate frequencies:
    out_f = calculate_frequencies_sprim(df, 'scn', relevant_evidence)
    outf = pd.DataFrame.from_dict(out_f, orient='index')
    outf.to_csv(f"{folder}/freq")

    #### BN learner #######
    arcs_scn, arcs_alt, scn_df, alt_df = arc_structure(sit_arg, df, evidence_attached)

    # add constraint node
    bn_scn = learn_bn(scn_df, arcs_scn, f"{folder}/scn")
    bn_alt = learn_bn(alt_df, arcs_alt, f"{folder}/alt")


    # print scores
    f = open(f"{folder}/scores_scn.txt", 'w')
    print(gumscores.computeScores(bn_scn, f"{folder}/df.csv")[1], file=f)

    f = open(f"{folder}/scores_alt.txt", 'w')
    print(gumscores.computeScores(bn_alt, f"{folder}/df.csv")[1], file=f)

    ####### inference with evidence # scn interpretation = conj



    out_scn, ev_prob = calculate_posterior_sprim(bn_scn, 'scn', relevant_evidence)
    out_alt, ev_prob = calculate_posterior_sprim(bn_alt, 'scn', relevant_evidence)

    outdf1 = pd.DataFrame.from_dict(out_scn, orient='index')
    outdf3 = pd.DataFrame.from_dict(out_alt, orient='index')

    plot_outcomes([outf, outdf1, outdf3], f"{folder}/")

    plot_outcomes_from_csv(f"{folder}/")


def param_sweep_situation(sit_arg, full_l):


    relevant_evidence = [{}, {"e1": 0}, {"e1": 1}, {"e2": 0}, {"e2": 1},
                         {"e1": 0, "e2": 0}, {"e1": 0, "e2": 1}, {"e1": 1, "e2": 0}, {"e1": 1, "e2": 1}]



    df = pd.DataFrame(map(Counter, full_l)).fillna(0, downcast='infer')
    cols = [c for c in df.columns if c[0] != "-"]
    df = df[cols]
    #df.to_csv(f"{folder}/df.csv", index=False)
    # visualize using venn diagrams


    #### calculate frequencies:
    out_f = calculate_frequencies_sprim(df, 'scn', relevant_evidence)
    outf = pd.DataFrame.from_dict(out_f, orient='index')

    #### BN learner #######
    arcs_scn, arcs_alt, scn_df, alt_df = arc_structure(sit_arg, df)

    bn_scn = learn_bn(scn_df, arcs_scn, "0")
    bn_alt = learn_bn(alt_df, arcs_alt, "0")

    ####### inference with evidence # scn interpretation = conj

    out_scn, ev_prob = calculate_posterior_sprim(bn_scn, 'scn', relevant_evidence)
    out_alt, ev_prob = calculate_posterior_sprim(bn_alt, 'scn', relevant_evidence)

    outdf1 = pd.DataFrame.from_dict(out_scn, orient='index')
    outdf3 = pd.DataFrame.from_dict(out_alt, orient='index')

    sig_dif = calculate_differences_params([outf, outdf1, outdf3])
    return sig_dif

def calculate_differences_params(outcome_list):

    outcomes = pd.concat(outcome_list, axis=1)
    outcomes["evidence"] = outcomes.index
    outcomes.columns = ["frequencyPQ", "scn", "conj", "evidence"]

    df_delta = pd.DataFrame()
    df_delta["scn"] = outcomes["frequencyPQ"] - outcomes["scn"]
    l = (df_delta["scn"] > 0.1).any().any()
    l1 = (df_delta["scn"] < -0.1).any().any()
    df_delta["conj"] = outcomes["frequencyPQ"] - outcomes["conj"]
    l2 = (df_delta["conj"] > 0.1).any().any()
    l3 = (df_delta["conj"] < -0.1).any().any()
    if l or l1 or l2 or l3:
        #print(outcomes)

        return outcomes[(outcomes["frequencyPQ"] - outcomes["scn"] > 0.1) | (outcomes["frequencyPQ"] - outcomes["scn"] < -0.1) |
        (outcomes["frequencyPQ"] - outcomes["conj"] > 0.1) | (outcomes["frequencyPQ"] - outcomes["conj"] < -0.1)]
    else:
        return "0"


def plot_outcomes(outcome_list, name):

    outcomes = pd.concat(outcome_list, axis=1)
    outcomes["evidence"] = outcomes.index
    outcomes.columns = ["frequencyPQ", "scn", "conj", "evidence"]
    outcomes.to_csv(name+"outcome.csv")

    outcomes.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=60)
    plt.title(name + "posterior")
    #ure(figsize=(20, 14))
    plt.savefig(f"{name}posteriors.png")
    #plt.show()

    df_delta = pd.DataFrame()
    df_delta["scn"] = outcomes["frequencyPQ"] - outcomes["scn"]
    df_delta["conj"] = outcomes["frequencyPQ"] - outcomes["conj"]
    df_delta.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=60)
    plt.title(name + "difference")
    #plt.figure(figsize=(20, 14))
    plt.savefig(f"{name}difference.png")
    #plt.show()

def plot_scenario_outcomes(outcome_list, name):

    outcomes = pd.concat(outcome_list, axis=1)

    print(outcomes)
    outcomes["evidence"] = outcomes.index
    flag = 0
    if len(outcomes.columns) < 5:
        outcomes.columns = ["frequencyscn1", "scn-scn1",  "conj-scn1",
                            "evidence", "outcomeProb"]
        flag = 1
    else:
        outcomes.columns = ["frequencyscn1", "frequencyscn2", "scn-scn1", "scn-scn2","outcomeProbScn", "conj-scn1", "conj-scn2", "outcomeProbConj", "evidence"]

    outcomes.to_csv(name+"outcome.csv")

    outcomes.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=60)
    plt.title(name + "posterior")
    #plt.figure(figsize=(20, 14))
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 13)
    plt.savefig(f"{name}posteriors.png")
    plt.show()

    outcome_scn1 = pd.DataFrame()
    outcome_scn1["fscn1"] = outcomes["frequencyscn1"]
    outcome_scn1["scn-scn1"] = outcomes["scn-scn1"]
    outcome_scn1["conj-scn1"] = outcomes["conj-scn1"]
    outcome_scn1.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=60)
    plt.title(name + "posterior scn1")
    #plt.figure(figsize=(20, 14))
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 13)
    plt.savefig(f"{name}posteriorsscn1.png")
    plt.show()

    if flag == 0:
        outcome_scn2 = pd.DataFrame()
        outcome_scn2["fscn2"] = outcomes["frequencyscn2"]
        outcome_scn2["scn-scn2"] = outcomes["scn-scn2"]
        outcome_scn2["conj-scn2"] = outcomes["conj-scn2"]
        outcome_scn2.plot(kind='bar')
        plt.subplots_adjust(bottom=0.3)
        plt.xticks(rotation=60)
        plt.title(name + "posterior scn2")
        #plt.figure(figsize=(20, 14))
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 13)
        plt.savefig(f"{name}posteriorsscn2.png")
        plt.show()

    '''df_delta = pd.DataFrame()
    df_delta["scn"] = outcomes["frequencyPQ"] - outcomes["scn"]
    df_delta["conj"] = outcomes["frequencyPQ"] - outcomes["conj"]
    df_delta.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=60)
    plt.title(name + "difference")
    plt.savefig(f"{name}difference.png")'''
    #plt.show()


def plot_outcomes_from_csv(outcome_path):
    df = pd.read_csv(f"{outcome_path}outcome.csv")

    dif_1 = pd.DataFrame()
    dif_1["dif1"] = df["frequencyscn1"] - df["conj-scn1"]
    dif_1["dif2"] = df["frequencyscn2"] - df["conj-scn2"]
    dif_1.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    #plt.figure(figsize=(20, 14))
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 13)
    plt.savefig(f"{outcome_path}/differencePosteriors_conj.png")
    plt.show()


    dif_2 = pd.DataFrame()
    dif_2["dif1"] = df["frequencyscn1"] - df["scn-scn1"]
    dif_2["dif2"] = df["frequencyscn2"] - df["scn-scn2"]
    dif_2.plot(kind='bar')
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    #plt.figure(figsize=(20, 14))
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 13)
    plt.savefig(f"{outcome_path}/differencePosteriors_scn.png")
    plt.show()



def compare_distributions(sit_arg):
    plot_distribution(f"situation{sit_arg}/df.csv", "conj")
    plot_distribution(f"situation{sit_arg}/df_sprim.csv", "scn")


def construct_string(row, columns):
    return ','.join([f'{col.lower()}={row[col]}' for col in columns])


def plot_distribution(df_csv, name):
    df_csv = pd.read_csv(df_csv)
    if name == "conj":
        df_csv = df_csv.iloc[:, [4, 2, 0, 1, 3]]
    print(df_csv["p"].value_counts())
    print(df_csv["q"].value_counts())
    print(df_csv["scn"].value_counts())

    print(df_csv[["q", "p"]].value_counts())
    print(df_csv[["q", "p", "scn"]].value_counts())

    print(df_csv.columns)
    '''d1 = df_csv[df_csv.columns].astype(str).apply(construct_string, axis=1, columns=df_csv.columns)
    d1.value_counts().plot(kind='bar', log=True)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(f"distribution_{name}")
    plt.show()'''
    # df_scn.plot(kind='bar')
    # plt.show()