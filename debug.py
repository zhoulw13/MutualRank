import pickle

with open('class.txt', 'rb') as f:
	runtest = pickle.load(f)


print ('Initialized.')
for i,item in enumerate(runtest.Items):
	runtest.TakeSamples(item, 100)
print ('Sampled.')

runtest.NormalW()
runtest.CalculateRank()
runtest.CalculateUncertainty()
runtest.OutputScore()

