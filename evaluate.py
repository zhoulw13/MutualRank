import os, json
from easydict import EasyDict

def evaluate(folder, topW, topI):
	dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
	static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
	manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

	workerOrder = sorted(range(len(dynamic_info.WorkerAccuracy)), key=lambda k: dynamic_info.WorkerAccuracy[k])
	print ('M3V Spammer Workers')
	M3VWorkers = [static_info.WorkerType[i] for i in workerOrder[0:topW]]
	print (M3VWorkers)

	print ('------------------')
	print ('Our Spammer Workers')
	with open('workerScore.txt', 'r') as f:
		WorkerScore = f.read().splitlines()
	workerOrder = sorted(range(len(WorkerScore)), key=lambda k: WorkerScore[k])
	OurWorkers = [static_info.WorkerType[i] for i in workerOrder[0:topW]]
	print (OurWorkers)

	print ('==================')

	print ('M3V Confusing Instances')
	InstanceScore = []
	label_list = []
	for i in range(manifest.InstanceTotalNum):
		label = dynamic_info.PosteriorDistribution[i].index(max(dynamic_info.PosteriorDistribution[i]))
		label_list.append(label)
		total = 0
		right = 0
		for j,l in enumerate(static_info.WorkerLabels[i]):
			if l != -1:
				total += dynamic_info.WorkerAccuracy[j]
				if l == label:
					right += dynamic_info.WorkerAccuracy[j]
		InstanceScore.append((right / total))
	instanceOrder = sorted(range(manifest.InstanceTotalNum), key=lambda k:InstanceScore[k])	
	M3VInstances = [abs(label_list[instanceOrder[i]]-static_info.true_labels[instanceOrder[i]]) for i in instanceOrder[0:topI]]
	print (M3VInstances)

	print ('------------------')
	print ('Our Confusing Instances')
	with open('instanceScore.txt', 'r') as f:
		instanceScore = f.read().splitlines()
	instanceOrder = sorted(range(manifest.InstanceTotalNum), key=lambda k: instanceScore[k])
	OurInstances = [abs(label_list[instanceOrder[i]]-static_info.true_labels[instanceOrder[i]]) for i in instanceOrder[0:topI]]
	print (OurInstances)

	print ('==================')

	a = '  '+str(topW-M3VWorkers.count(1))+'/'+str(topW)
	b = '  '+str(topW-OurWorkers.count(1))+'/'+str(topW)
	c = '   '+str(topI-M3VInstances.count(0))+'/'+str(topI)
	d = '   '+str(topI-OurInstances.count(0))+'/'+str(topI)
	print ('|----------|----------|----------|')
	print ('|          |   M3V    |    Our   |')
	print ('|----------|----------|----------|')
	print ('|  Worker  |'+a.ljust(10)+'|'+b.ljust(10)+'|')
	print ('|----------|----------|----------|')
	print ('| Instance |'+c.ljust(10)+'|'+d.ljust(10)+'|')
	print ('|----------|----------|----------|')
