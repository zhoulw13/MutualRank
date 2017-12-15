from MutualRank import MutualRank
from dataOrganize import dataOrganize

data = dataOrganize('info')
print (data.InstanceNN[33].Neighbors[33])

#data = EasyDict(json.load(open("test.json")))
#runTest = MutualRank(data)
#runTest.Run()