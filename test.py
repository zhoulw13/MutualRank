from MutualRank import MutualRank
import json
from easydict import EasyDict


data = EasyDict(json.load(open("test.json")))

runTest = MutualRank(data)
runTest.Run()