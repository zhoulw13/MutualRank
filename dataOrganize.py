import os, json
from easydict import EasyDict

def dataOrganize(folder):
	if os.path.isfile(folder+'.json'):
		return EasyDict(json.load(open(folder+'.json')))

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
			'Quality': 1,
			'Uncertainty': dynamic_info.Uncertainty[i],
		}))
		data.InstanceNN.append(EasyDict({
			'Index': i,
			'Neighbors': []
		}))

	# labeling relation
	for i in range(data.InstanceCount):
		posterior = dynamic_info.PosteriorDistribution[i].index(max(dynamic_info.PosteriorDistribution[i]))
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


	# instance quality
	for i in range(data.InstanceCount):
		label = dynamic_info.PosteriorDistribution[i].index(max(dynamic_info.PosteriorDistribution[i]))
		total = 0
		right = 0
		for j,l in enumerate(static_info.WorkerLabels[i]):
			if l != -1:
				total += data.Workers[j].Quality
				if l == label:
					right += data.Workers[j].Quality
		data.Instances[i].Quality = right / total
	
	# instance similarity graph
	index = 0
	for i in range(data.InstanceCount):
		for j in range(i+1,data.InstanceCount):
			similarity = static_info.SimiGraph[index]
			data.InstanceNN[i].Neighbors.append(EasyDict({
				'Index': j,
				'Similarity': similarity
			}))
			data.InstanceNN[j].Neighbors.append(EasyDict({
				'Index': i,
				'Similarity': similarity
			}))
			index += 1

	# worker similarity graph
	for i in range(data.WorkerCount):
		for j in range(i+1, data.WorkerCount):
			intersection = set(data.Workers[j].Instances).intersection(data.Workers[i].Instances)
			if len(intersection) == 0:
				continue
			sameChoice = 0
			for item in intersection:
				if static_info.WorkerLabels[item][i] == static_info.WorkerLabels[item][j]:
					sameChoice += 1
			similarity = sameChoice / len(intersection)
			data.WorkerNN[i].Neighbors.append(EasyDict({
				'Index': j,
				'Similarity': similarity
			}))
			data.WorkerNN[j].Neighbors.append(EasyDict({
				'Index': i,
				'Similarity': similarity
			}))

	with open(folder+'.json', 'w') as f:
		json.dump(data, f)
	return data