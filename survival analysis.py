import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# load dataset
data = pd.read_csv("lung.csv")
# print(data.head())
# print(data.columns)
# data.info()
# print(data.describe())

# sex distribution histogram
# print(data["sex"].hist())
# plt.show()

kmf= KaplanMeierFitter() # Create an object for Kaplan-Meier-Fitter

#Organize the data
data.loc[data["status"]==1, "dead"]=0 # if status is 1 then dead =0
data.loc[data["status"]==2, "dead"]=1 # (status 1 - censored, 2 - dead )
# print(data.head())

# Fitting our data into an object
kmf.fit(data["time"],data["dead"])
kmf.plot() # plot diagram
# plt.title('Kaplan-Meier Estimate')
# plt.ylabel('Probability 1-Alive 0-Dead')
# plt.xlabel('Number of Days')
# plt.show()

print(kmf.event_table) #Generate event table

######## Survival probability at t=0 only
event_at_0= kmf.event_table.iloc[0,:]
survival_for_0= (event_at_0.at_risk - event_at_0.observed)/ event_at_0.at_risk
print("Surival probability at time 0 only is : ", survival_for_0)

######## Survival probability at t=5 only
event_at_5= kmf.event_table.iloc[1,:]
survival_for_5= (event_at_5.at_risk - event_at_5.observed)/ event_at_5.at_risk
print("Surival probability at time 5 only is : ", survival_for_5)

######## Survival probability at t=13 only
event_at_13= kmf.event_table.iloc[4,:]
survival_for_13= (event_at_13.at_risk - event_at_13.observed)/ event_at_13.at_risk
print("Surival probability at time 13 only is : ", survival_for_13)

##### Survival probability probability after 5 days (for t= 5)
survival_after_5= survival_for_0 * survival_for_5
print("\nSurvival Probability after 5 days : ", survival_after_5)

#### Automate the work we've done above
print("\nSurvival Probability after 5 days : ",kmf.predict(5) )
print("Survival Probability after 3 days : ",kmf.predict(13) )
print("Survival Probability after 1022 days : ",kmf.predict(1022) )

#### Survival probability for whole timeline
print("\n",kmf.survival_function_)
