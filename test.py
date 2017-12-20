from MutualRank import MutualRank
from dataOrganize import dataOrganize
from evaluate import evaluate

evaluate('info', 40, 40)

#data = dataOrganize('info')
#print ("Data Loaded.")

'''import os, json
from easydict import EasyDict
folder = 'info'
dynamic_info = EasyDict(json.load(open(os.path.join(folder,'dynamic_info.json'))))
static_info = EasyDict(json.load(open(os.path.join(folder,'static_info.json'))))
manifest = EasyDict(json.load(open(os.path.join(folder,'manifest.json'))))

print (data.WorkerNN[0].Neighbors)'''

#data = EasyDict(json.load(open("test.json")))
#runTest = MutualRank(data)
#runTest.Run()