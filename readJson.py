import json
import statistics
from statistics import mode



with open("data.json","r") as f:
    #load the file into a variable
    data=json.load(f)

print(len(data))
#print(data)
#for key,value in data:
 #      print(f"{key}: {value}")

model_type_list=[]
negative_sampling_list=[]
n_neighbours_list=[]
epsilon_list=[]
n_epochs_list=[]
lr_list=[]
proportion_list=[]
for i in range(len(data)):
        model_type_list.append(data[i]['config']['model_type'])
        negative_sampling_list.append(data[i]['config']['negative_sampling'])
        n_neighbours_list.append(data[i]['config']['n_neighbours'])
        epsilon_list.append(data[i]['config']['epsilon'])
        n_epochs_list.append(data[i]['config']['n_epochs'])
        lr_list.append(data[i]['config']['lr'])
        proportion_list.append(data[i]['config']['proportion'])
#average = sum_of_list/len(data)


print("Mode of model_type is", mode(model_type_list))
print("Mode of negative_sampling is", mode(negative_sampling_list))
print("Mean of n_neighbours is", statistics.mean(n_neighbours_list))
print("Median of n_neighbours is", statistics.median(n_neighbours_list))
print("Mode of n_neighbours is", mode(n_neighbours_list))
print("Mean of number of epochs to be used in training NN is", statistics.mean(n_epochs_list))
print("Mode of epochs is", mode(n_epochs_list))
print("Median of epochs is", statistics.median(n_epochs_list))
print("Mean of learning rate is", statistics.mean(lr_list))
print("Mode of lr is", mode(lr_list))
print("Median of lr is", statistics.median(lr_list))
print("Mean of epsilon used in negative sampling is", statistics.mean(epsilon_list))
print("Median of epsilon is", statistics.median(epsilon_list))
print("Mode of epsilon is", mode(epsilon_list))
print("Mean of proportion used in negative sampling is", statistics.mean(proportion_list))
print("Median of proportion is", statistics.median(proportion_list))
print("Mode of n_neighbours is", mode(proportion_list))
