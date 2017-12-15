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
			'Index': i+1,
			'Instances': [],
			'Quality': dynamic_info.WorkerAccuracy[i],
			'Uncertainty': 1,
		}))
		data.WorkerNN.append(EasyDict({
			'Index': i+1,
			'Neighbors': []
		}))

	for i in range(data.InstanceCount):
		data.Instances.append(EasyDict({
			'Type': 1,
			'Index': i+1,
			'Workers': [],
			'Quality': 1,
			'Uncertainty': dynamic_info.Uncertainty[i],
		}))
		data.InstanceNN.append(EasyDict({
			'Index': i+1,
			'Neighbors': []
		}))

	# labeling relation
	for i in range(data.InstanceCount):
		for j,label in enumerate(static_info.WorkerLabels[i]):
			if label != -1:
				data.Workers[j].Instances.append(i)
				data.Instances[i].Workers.append(j)
	
	# instance similarity graph
	index = 0
	for i in range(data.InstanceCount):
		for j in range(i+1,data.InstanceCount):
			similarity = static_info.SimiGraph[index]
			data.InstanceNN[i].Neighbors.append(EasyDict({
				'Index': j+1,
				'Similarity': similarity
			}))
			data.InstanceNN[j].Neighbors.append(EasyDict({
				'Index': i+1,
				'Similarity': similarity
			}))
			index += 1

	with open(folder+'.json', 'w') as f:
		json.dump(data, f)
	return data