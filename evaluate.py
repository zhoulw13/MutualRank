import os, json, sys
from easydict import EasyDict

def evaluate(folder, topW, topI):
	dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
	static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
	manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

	print ('==================')
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
	label_list = []
	for i in range(manifest.InstanceTotalNum):
		label_list.append(dynamic_info.PosteriorDistribution[i].index(max(dynamic_info.PosteriorDistribution[i])))
	'''InstanceScore = []
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
	instanceOrder = sorted(range(manifest.InstanceTotalNum), key=lambda k:InstanceScore[k])	'''
	instanceOrder = sorted(range(len(dynamic_info.Uncertainty)), key=lambda k: -dynamic_info.Uncertainty[k])
	M3VInstances = [abs(label_list[instanceOrder[i]]-static_info.true_labels[instanceOrder[i]]) for i in instanceOrder[0:topI]]
	print (M3VInstances)

	print ('------------------')
	print ('Our Confusing Instances')
	with open('instanceUncertainty.txt', 'r') as f:
		instanceScore = f.read().splitlines()
	instanceOrder = sorted(range(manifest.InstanceTotalNum), key=lambda k: instanceScore[k], reverse=True)
	OurInstances = [abs(label_list[instanceOrder[i]]-static_info.true_labels[instanceOrder[i]]) for i in instanceOrder[0:topI]]
	print (OurInstances)

	print ('==================')

	a = '  '+str(M3VWorkers.count(3)+M3VWorkers.count(4))+'/'+str(topW)
	b = '  '+str(OurWorkers.count(3)+OurWorkers.count(4))+'/'+str(topW)
	c = '   '+str(topI-M3VInstances.count(0))+'/'+str(topI)
	d = '   '+str(topI-OurInstances.count(0))+'/'+str(topI)
	print ('|----------|----------|----------|')
	print ('|          |   M3V    |    Our   |')
	print ('|----------|----------|----------|')
	print ('|  Worker  |'+a.ljust(10)+'|'+b.ljust(10)+'|')
	print ('|----------|----------|----------|')
	print ('| Instance |'+c.ljust(10)+'|'+d.ljust(10)+'|')
	print ('|----------|----------|----------|')

if __name__ == "__main__":
	if len(sys.argv) == 2: # folder
		evaluate(sys.argv[1], 10, 10)
	elif len(sys.argv) == 3: # topW topI
		evaluate('info', int(sys.argv[1]), int(sys.argv[2]))
	elif len(sys.argv) == 4: # folder topW topI
		evaluate(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
	else:
		evaluate('info', 10, 10)
