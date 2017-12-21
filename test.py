from MutualRank import MutualRank
from dataReorganize import dataReorganize
import pickle

data = dataReorganize('info')
print ("Data Loaded.")

'''
import os, json
from easydict import EasyDict
folder = 'info'
dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

#for i in range(data.InstanceCount):
#	if data.Instances[i].Match.count(1) == 0:
#		print ('wubba lubba dub dub')
for i in range(data.WorkerCount):
	if len(data.WorkerNN[i].Neighbors) == 0:
		print ('wubba lubba dub dub')
	#print (len(data.Instance[i].Neighbors), static_info.WorkerType[i])
'''


#data = EasyDict(json.load(open("test.json")))
runTest = MutualRank(data)
runTest.Run()