import os, json
from easydict import EasyDict
import numpy as np
from sklearn.cluster import KMeans

def similarityCal(folder):
	dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
	static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
	manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

	data = EasyDict()
	data.WorkerCount = manifest.WorkerTotalNum
	data.InstanceCount = manifest.InstanceTotalNum
	data.Workers = []
	data.Instances = []
	data.WorkerNN = []
	data.InstanceNN = []

	similarity = np.zeros((data.WorkerCount, data.WorkerCount))

	dynamic_info.PosteriorDistribution = np.array(dynamic_info.PosteriorDistribution).reshape(manifest.InstanceTotalNum, 4)
	static_info.WorkerLabels = np.array(static_info.WorkerLabels).reshape(manifest.InstanceTotalNum, manifest.WorkerTotalNum)

	for i in range(data.WorkerCount):
		data.Workers.append(EasyDict({
			'Type': 0,
			'Index': i,
			'Instances': [],
			'Match': [],
			'Quality': dynamic_info.WorkerAccuracy[i],
			'Uncertainty': 1,
		}))
		data.WorkerNN.append(EasyDict({
			'Index': i,
			'Neighbors': []
		}))

	for i in range(data.InstanceCount):
		data.Instances.append(EasyDict({
			'Type': 1,
			'Index': i,
			'Workers': [],
			'Match': [],
			'Quality': 1,#-dynamic_info.Uncertainty[i]/max(dynamic_info.Uncertainty),
			'Uncertainty': dynamic_info.Uncertainty[i],
		}))
		data.InstanceNN.append(EasyDict({
			'Index': i,
			'Neighbors': []
		}))

	# labeling relation
	for i in range(data.InstanceCount):
		posterior = np.argmax(dynamic_info.PosteriorDistribution[i])
		for j,label in enumerate(static_info.WorkerLabels[i]):
			if label != -1:
				data.Workers[j].Instances.append(i)
				data.Instances[i].Workers.append(j)
				if label == posterior:
					data.Workers[j].Match.append(1)
					data.Instances[i].Match.append(1)
				else:
					data.Workers[j].Match.append(-1)
					data.Instances[i].Match.append(-1)

	for i in range(data.WorkerCount):
		for j in range(i+1, data.WorkerCount):
			intersection = set(data.Workers[j].Instances).intersection(data.Workers[i].Instances)
			sameChoice = 0
			for item in intersection:
				posterior = np.argmax(dynamic_info.PosteriorDistribution[item])
				if static_info.WorkerLabels[item][i] == static_info.WorkerLabels[item][j]:
					similarity[i][j] += 1 / np.log((np.sum(static_info.WorkerLabels[item] == static_info.WorkerLabels[item][i])))

			#if len(intersection) == 0 or sameChoice == 0:
			#	continue
			
			#similarity[i][j] = sameChoice / len(intersection)
			similarity[j][i] = similarity[i][j]

	#return similarity, 1

	cms = []
	conditionNum = []
	for i in range(data.WorkerCount):
		confusion_matrix = np.zeros((4,4))
		for instance in data.Workers[i].Instances:
			posterior = np.argmax(dynamic_info.PosteriorDistribution[instance])
			#print(instance, posterior, dynamic_info.PosteriorDistribution[instance])
			choice = static_info.WorkerLabels[instance][i] 

			confusion_matrix[choice][posterior] += 1

		confusion_matrix = (confusion_matrix+1) / np.sum(confusion_matrix+1, axis=0)
		cms.append(confusion_matrix)
		conditionNum.append(np.linalg.cond(confusion_matrix))

	#print(cms[1])
	#print(cms[2])
	'''
	for i in range(data.WorkerCount):
		confusion_matrix = cms[i]
		a = confusion_matrix[0,1]+confusion_matrix[1,0]
		b = confusion_matrix[2,3]+confusion_matrix[3,2]
		c = 0
		for j in range(4):
			for k in range(j+1, 4):
				c += confusion_matrix[j,k]+confusion_matrix[k,j]
		print(i, a, b, c)
	'''

	#print (np.array(cms).shape)
	#kmeans = KMeans(n_clusters=4, random_state=0).fit(np.array(cms))
	#print (kmeans.labels_)
	#for i in range(4):
	#	print ([j for j,x in enumerate(kmeans.labels_) if x == i])
		#print (i, static_info.WorkerType[i], np.linalg.cond(confusion_matrix))
	#print  ([static_info.WorkerType[x] for x in sorted(range(data.WorkerCount), key=lambda x:conditionNum[x])])
	#for i in [57, 28, 1, 2, 25, 30, 13, 55, 27, 34, 15, 21, 9, 36, 14]:
	#	print (cms[i], i)
		

	for i in range(data.WorkerCount):
		for j in range(i+1, data.WorkerCount):
			similarity[i][j]  = 1 / (np.sum(np.square(cms[i]-cms[j])))
			similarity[j][i] = similarity[i][j]


	return similarity, conditionNum #, static_info.WorkerType


def confusionMatrixClustering(k, confusionMatrixs):
	length = len(confusionMatrixs)
	kmeans = KMeans(n_clusters=4, random_state=0).fit(confusionMatrixs)


import json

if __name__ == '__main__':
	similarity, conditionNum = similarityCal('real')

	'''
	x = {}
	x['nodes'] = []
	for i in range(len(similarity)):
		x['nodes'].append({'id': i, 'group': 0})
	for i in [57, 28, 2, 25, 30, 13]:
		x['nodes'][i]['group'] = 1
	for i in [1, 36, 14]:
		x['nodes'][i]['group'] = 2
	for i in [55, 27, 34, 15, 21, 9]:
		x['nodes'][i]['group'] = 3

	x['links'] = []
	for i in range(len(similarity)):
		for j in range(i+1, len(similarity)):
			if similarity[i][j] > 10:
				x['links'].append({'source': i, 'target': j, 'value': similarity[i][j]/10})

	with open('data2.json', 'w') as fp:
		json.dump(x, fp, sort_keys=True, indent=4)
	'''

	take_num = 5
	for i in range(len(similarity)):
		print(sorted(similarity[i], reverse=True)[:take_num])


	take_num = 3
	sums = 0
	
	for i in range(len(similarity)):
		neighbor = sorted(range(len(similarity)), key=lambda x: similarity[i][x], reverse=True)[:take_num]
		#print(i, neighbor)
		#print (workerType[i], [workerType[x] for x in neighbor]
		if i in [57, 28, 1, 2, 25, 30, 13, 55, 27, 34, 15, 21, 9, 36, 14]:
			print (i, neighbor)
		#sums += sum([workerType[x] == workerType[i] for x in neighbor])

	#print (sums/take_num/len(similarity))
