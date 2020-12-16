import pandas as pd
from apyori import apriori
store_data=pd.read_csv('E:/Projects/Python/Big data/groceries - groceries.csv')
print(store_data.head())
print("Shape",store_data.shape)
#store_data=store_data.dropna() #remove nan
records=[]
for i in range(0, 22):
    records.append([str(store_data.values[i,j]) for j in range(1, 15)])
association_rules =apriori(records,min_support=0.02,min_confidence=0.5,min_lift=10,min_length=2)
association_results = list(association_rules)
print("Length of association rule : ",len(association_results))
print("Association result: ",association_results[0])
#result_list=[]
for item in association_results:
	pair=item[0]
	items=[x for x in pair]
	print("Rule: "+items[0] +"->" +items[1])
	print("Support: "+str(item[1]))
	print("Confidence: "+str(item[2][0][2]))
	print("Lift: "+str(item[2][0][3]))
	print("-------------------------------")
# print(len(association_results))
# print(association_results)

