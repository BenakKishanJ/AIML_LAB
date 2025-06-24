import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('dataset/sports.csv'))
print('The Dataset is: \n')
print(data)
concepts=np.array(data.iloc[:,0:-1])
print('The Concepts are: \n',concepts)
target=np.array(data.iloc[:,-1])
print('The Target is: \n',target)

def candidate_elimination(concepts, target):
    # Initialize specific and general boundaries
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, val in enumerate(target):
        if val == "Yes":
            for x in range(len(specific_h)):
                if concepts[i][x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        else:
            for x in range(len(specific_h)):
                if concepts[i][x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]

    # Remove duplicates or overly general hypotheses
    general_h = [gh for gh in general_h if gh != ["?" for _ in range(len(specific_h))]]

    return specific_h, general_h

s_final,g_final=candidate_elimination(concepts,target)
print('The Specific Concepts are: \n',s_final)
print('The General Concepts are: \n',g_final)
