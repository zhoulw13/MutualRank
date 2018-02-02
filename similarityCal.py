import os, json
from easydict import EasyDict
import numpy as np

def similarityCal(folder):
	dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
	static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
	manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

	data = EasyDict()
	data.WorkerCount = manifest.WorkerTotalNum
	data.InstanceCount = manifest.InstanceTotalNum
	data.Workers = []
	data.Instances = []

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

	for i in range(data.InstanceCount):
		data.Instances.append(EasyDict({
			'Type': 1,
			'Index': i,
			'Workers': [],
			'Match': [],
			'Quality': 1,
			'Uncertainty': dynamic_info.Uncertainty[i],
		}))

	# labeling relation
	for i in range(data.InstanceCount):
		for j,label in enumerate(static_info.WorkerLabels[i]):
			if label != -1:
				data.Workers[j].Instances.append(i)
				data.Instances[i].Workers.append(j)

	for i in range(data.WorkerCount):
		for j in range(i+1, data.WorkerCount):
			intersection = set(data.Workers[j].Instances).intersection(data.Workers[i].Instances)
			sameChoice = 0
			for item in intersection:
				if static_info.WorkerLabels[item][i] == static_info.WorkerLabels[item][j]:
					similarity[i][j] += 1 / np.log(np.sum(static_info.WorkerLabels[item] == static_info.WorkerLabels[item][i]))
			similarity[j][i] = similarity[i][j]

	cms = []
	for i in range(data.WorkerCount):
		confusionMatrix = np.zeros((4,4))
		for instance in data.Workers[i].Instances:
			posterior = np.argmax(dynamic_info.PosteriorDistribution[instance])
			choice = static_info.WorkerLabels[instance][i] 

			confusionMatrix[choice][posterior] += 1

		confusionMatrix = (confusionMatrix+1) / np.sum(confusionMatrix+1, axis=0)
		cms.append(confusionMatrix)

	mix = []
	for i in range(data.WorkerCount):
		confusionMatrix = cms[i]
		a = confusionMatrix[0,1]+confusionMatrix[1,0]
		b = confusionMatrix[2,3]+confusionMatrix[3,2]
		c = confusionMatrix[0,0]+confusionMatrix[1,1]
		d = confusionMatrix[2,2]+confusionMatrix[3,3]
		x = np.array([d*a, c*b, a*b, c*d])
		mix.append(x)
	mix = np.array(mix)

	for i in range(data.WorkerCount):
		for j in range(i+1, data.WorkerCount):
			similarity[i][j] *= 1 / (0.1+np.sum(np.square(mix[i]-mix[j])))
			similarity[j][i] = similarity[i][j]


	return similarity


def evaluate(neighbors):
	a = [57, 28, 2, 25, 30, 13]
	b = [1, 36, 14]
	c = [55, 27, 34, 15, 21, 9]
	hit = [0]*(len(a+b+c))

	ind = 0
	for spammer in a:
		for neighbor in neighbors[spammer]:
			if neighbor in a or neighbor in b or neighbor in c:
				hit[ind] += 1
		ind += 1
	for spammer in b:
		for neighbor in neighbors[spammer]:
			if neighbor in a or neighbor in b:
				hit[ind] += 1
		ind += 1
	for spammer in c:
		for neighbor in neighbors[spammer]:
			if neighbor in a or neighbor in c:
				hit[ind] += 1
		ind += 1


	print(hit, str(sum(hit))+'/'+str(len(hit)*len(neighbors[0])))

if __name__ == '__main__':
	similarity = similarityCal('real')

	takeNum = 5
	neighbors = []
	
	for i in range(len(similarity)):
		neighbor = sorted(range(len(similarity)), key=lambda x: similarity[i][x], reverse=True)[:takeNum]
		neighbors.append(neighbor)

	evaluate(neighbors)

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

