import os, json
from easydict import EasyDict

def evaluate(folder, topW, topI):
	dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
	static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
	manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

	workerOrder = sorted(range(len(dynamic_info.WorkerAccuracy)), key=lambda k: dynamic_info.WorkerAccuracy[k])
	print ('M3V Worker Result')
	for i in range(topW):
		print (static_info.WorkerType[workerOrder[i]])

	print ('------------------')
	print ('Our Worker Result')
	with open('workerScore.txt', 'r') as f:
		WorkerScore = f.read().splitlines()
	workerOrder = sorted(range(len(WorkerScore)), key=lambda k: WorkerScore[k])
	for i in range(topW):
		print (static_info.WorkerType[workerOrder[i]])