import numpy as np
from .models import ProbabilitySparseMatrix
from .models import Path
import random

class MutualRank:

	def __init__(self, data):
		self.data = data

	def Initialize(self):
		#initialize variables
		self.Wall = np.array([[0.8,0.2],[0.8,0.2]], dtype=np.double)
		
		self.WorkerTypeId = 0
		self.InstanceTypeId = 1

		self.Z = np.array([np.array([{} for i in range(self.data.WorkerCount)], dtype=np.object), np.array([{} for i in range(self.data.InstanceCount)], dtype=np.object)], dtype=np.object)
		self.M = np.array([np.zeros(self.data.WorkerCount, dtype=np.int), np.zeros(self.data.InstanceCount, dtype=np.int)], dtype=np.object)
		self.W = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)
		self.Q = np.array([np.zeros(self.data.WorkerCount, dtype=np.float), np.zeros(self.data.InstanceCount, dtype=np.float)], dtype=np.object)
		self.Q_sum = np.array([np.zeros(self.data.WorkerCount, dtype=np.float), np.zeros(self.data.InstanceCount, dtype=np.float)], dtype=np.object)

		self.SumW = 0

		self.NormalizeValue = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)
		self.NormalizeValueWeight = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)
		self.NormalizeValueToWorker = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)
		self.NormalizeValueToInstance = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)

		self.V = np.array([np.empty(self.data.WorkerCount, dtype=np.float), np.empty(self.data.InstanceCount, dtype=np.float)], dtype=np.object)

		self.WorkerUncertaintyMax = -1
		self.WorkerUncertaintyMin = -1
		self.InstanceUncertaintyMax = -1
		self.InstanceUncertaintyMin = -1

		self.HaveModedItems = []

		self.R = 1 # num of RW samples
		self.C = 0.8 # threshold of skipping RW 

		#type: List of Path
		self.InvertedItem2Path = np.array([[[] for i in range(self.data.WorkerCount)], [[]for i in range(self.data.InstanceCount)]], dtype=np.object)

		#initialize ...
		self.Worker2Instance = ProbabilitySparseMatrix(self.data.WorkerCount, self.data.InstanceCount)
		self.Worker2Worker = ProbabilitySparseMatrix(self.data.WorkerCount, self.data.WorkerCount)

		self.Instance2Instance = ProbabilitySparseMatrix(self.data.InstanceCount, self.data.InstanceCount)
		self.Instance2Worker = ProbabilitySparseMatrix(self.data.InstanceCount, self.data.WorkerCount)

		for worker in sorted(self.data.Workers, key=lambda x: x.Index, reverse=False):
			for index in sorted(worker.Instances, key=lambda x: x, reverse=False):
				instance = self.data.Instances[index]
				self.Worker2Instance.AddItem(worker, instance, instance.Quality)

			neighbors = filter(lambda x: x.Index == worker.Index, self.data.WorkerNN)[0].Neighbors
			for neighborPair in sorted(neighbors, key=lambda x: x.Index, reverse=False):
				neighbor = filter(lambda x: x.Index == neighborPair.Index, self.data.Workers)[0]
				self.Worker2Worker.AddItem(worker, neighbor, neighbor.Quality)

		for instance in sorted(self.data.Instances, key=lambda x: x.Index, reverse=False):
			for index in sorted(instance.Workers, key=lambda x: x, reverse=False):
				worker = self.data.Workers[index]
				self.Instance2Worker.AddItem(instance, worker, worker.Quality)

			neighbors = filter(lambda x: x.Index == instance.Index, self.data.InstanceNN)[0].Neighbors
			for neighborPair in sorted(neighbors, key=lambda x: x.Index, reverse=False):
				neighbor = filter(lambda x: x.Index == neighborPair.Index, self.data.Instances)[0]
				self.Instance2Instance.AddItem(instance, neighbor, neighbor.Quality)

		self.Worker2Instance.GetSumValue()
		self.Worker2Worker.GetSumValue()

		self.Instance2Instance.GetSumValue()
		self.Instance2Worker.GetSumValue()

		self.Items = []
		for worker in self.data.Workers:
			self.Items.append(worker)
		for instance in self.data.Instances:
			self.Items.append(instance)
		self.N = len(self.Items)

		for item in self.Items:
			typeId, index = self.GetTupleIndex(item)

			self.W[typeId][index] = item.Quality

			self.GetNormalizeValue(typeId, index)

			pi = 0.0
			if typeId == self.WorkerTypeId:
				pi = self.Worker2Worker.SrcSumValue[index]*self.Wall[0,0]+self.Worker2Instance.SrcSumValue[index]*self.Wall[0,1]
			elif typeId == self.InstanceTypeId:
				pi = self.Instance2Worker.SrcSumValue[index]*self.Wall[1,0]+self.Instance2Instance.SrcSumValue[index]*self.Wall[1,1]
			self.NormalizeValue[typeId][index] = pi


	def GetTupleIndex(self, item):
		if item.Type == self.WorkerTypeId:
			return (self.WorkerTypeId, item.Index)
		elif item.Type == self.InstanceTypeId:
			return (self.InstanceTypeId, item.Index)
		return None

	def GetItemByTupleIndex(self, key):
		typeId, index = key
		if typeId == self.WorkerTypeId:
			return self.data.Workers[index]
		elif typeId == self.InstanceTypeId:
			return self.data.Instances[index]
		return None

	def GetNormalizeValue(self, typeId, index):
		if typeId == self.WorkerTypeId:
			self.NormalizeValueToWorker[typeId][index] = self.Worker2Worker.SrcSumValue[index]
			self.NormalizeValueToInstance[typeId][index] = self.Worker2Instance.SrcSumValue[index]
			self.NormalizeValueWeight[typeId][index] = 0
			if self.NormalizeValueToWorker[typeId][index] != 0:
				self.NormalizeValueWeight[typeId][index] += self.Wall[0,0]
			if self.NormalizeValueToInstance[typeId][index] != 0:
				self.NormalizeValueWeight[typeId][index] += self.Wall[0,1]
		elif typeId == self.InstanceTypeId:
			self.NormalizeValueToWorker[typeId][index] = self.Instance2Worker.SrcSumValue[index]
			self.NormalizeValueToInstance[typeId][index] = self.Instance2Instance.SrcSumValue[index]
			self.NormalizeValueWeight[typeId][index] = 0
			if self.NormalizeValueToWorker[typeId][index] != 0:
				self.NormalizeValueWeight[typeId][index] += self.Wall[1,0]
			if self.NormalizeValueToInstance[typeId][index] != 0:
				self.NormalizeValueWeight[typeId][index] += self.Wall[1,1]

	def TakeSamples(self, item, d):
		for i in range(int(d)):
			path = self.ConstructPath(item)
			self.UpdatePathStat(path)
			for typeId, index in path.Samples:
				self.InvertedItem2Path[typeId][index].append(path)


	def ConstructPath(self, item):
		samples = []
		while item != None:
			samples.append(self.GetTupleIndex(item))
			item = self.GetNextStep(item)
		return Path(samples)

	def GetNextStep(self, item):
		nextSkip = random.uniform(0,1)
		if nextSkip > self.C:
			return None
		elif item.Type == self.WorkerTypeId:
			nextType = random.uniform(0,1)*(np.sum(self.Wall, axis=0)[0])
			findGoodSample = False
			while not findGoodSample:
				if nextType < self.Wall[0,0]:
					if len(self.Worker2Worker.Srcs[item.Index]) != 0:
						findGoodSample = True
				else:
					if len(self.Worker2Instance.Srcs[item.Index]) != 0:
						findGoodSample = True
				if not findGoodSample:
					nextType = random.uniform(0,1)*(np.sum(self.Wall, axis=0)[0])
			# found one
			if nextType < self.Wall[0,0]:
				if len(self.Worker2Worker.Srcs[item.Index]) != 0:
					return self.SampleFromMultinomial(self.Worker2Worker, item.Index)
				return None
			else:
				if len(self.Worker2Instance.Srcs[item.Index]) != 0:
					return self.SampleFromMultinomial(self.Worker2Instance, item.Index)
				return None
		elif item.Type == self.InstanceTypeId:
			nextType = random.uniform(0,1)*(np.sum(self.Wall, axis=0)[1])
			findGoodSample = False
			while not findGoodSample:
				if nextType < self.Wall[1,0]:
					if len(self.Instance2Worker.Srcs[item.Index]) != 0:
						findGoodSample = True
				else:
					if len(self.Instance2Instance.Srcs[item.Index]) != 0:
						findGoodSample = True
				if not findGoodSample:
					nextType = random.uniform(0,1)*(np.sum(self.Wall, axis=0)[1])
			# found one
			if nextType < self.Wall[1,0]:
				if len(self.Instance2Worker.Srcs[item.Index]) != 0:
					return self.SampleFromMultinomial(self.Instance2Worker, item.Index)
				return None
			else:
				if len(self.Instance2Instance.Srcs[item.Index]) != 0:
					return self.SampleFromMultinomial(self.Instance2Instance, item.Index)
				return None

	def SampleFromMultinomial(self, mat, srcIndex):
		sums, items = mat.ForSample[srcIndex]

		nextItem = random.uniform(0,1)*sums[-1]
		for i in range(len(sums)):
			if nextItem < sums[i]:
				return items[i]

	def UpdatePathStat(self, path):
		for i in range(len(path.Samples)):
			typeId, index = path.Samples[i]
			self.M[typeId][index] += 1
			self.Q_sum[typeId][index] += 1
			qb = False
			for j in range(i, len(path.Samples)):
				vpass = path.Samples[j]
				if vpass in self.Z[typeId][index]:
					self.Z[typeId][index][vpass] += 1
				else:
					self.Z[typeId][index][vpass] = 1
				if vpass == (typeId, index) and j > i:
					qb = True
			if qb:
				self.Q[typeId][index] += 1

	def NormalW(self):
		self.SumW = np.sum(self.W[0])+np.sum(self.W[1])

	def CalculateRank(self):
		for item in self.Items:
			item.Score = 0
		for i in range(len(self.Z)):
			for j in range(len(self.Z[i])):
				for key, value in self.Z[i][j].iteritems():
					item = self.GetItemByTupleIndex(key)
					item.Score += (1-self.C) * self.W[i][j] * value / self.M[i][j] / self.SumW

	def CalculateUncertainty(self):
		for i in range(len(self.Z)):
			for j in range(len(self.Z[i])):
				for key, value in self.Z[i][j].iteritems():
					typeId, index = key
					EZij = value / self.M[i][j]
					qjj = 1 - 1.0 / self.Z[typeId][index][key] * self.M[typeId][index]
					VZij = (1 + qjj) / (1 - qjj) * EZij - EZij * EZij
					self.V[typeId][index] += pow(self.W[i][j], 2) / self.M[i][j] * VZij / pow(self.SumW, 2)

		for item in self.Items:
			tupleIndex = self.GetTupleIndex(item)
			# item moded to be add
			if item in self.HaveModedItems:
				item.Uncertainty = 0.1
			else:
				typeId, index = tupleIndex
				VMR = self.V[typeId][index] / item.Score
				if VMR < 0:
					item.Uncertainty = 0
				else:
					item.Uncertainty = VMR

		# initialize bound
		if self.WorkerUncertaintyMax == -1:
			sortedWorker = sorted(self.data.Workers, key=lambda x: -x.Uncertainty, reverse=True)
			self.WorkerUncertaintyMin = sortedWorker[-1].Uncertainty
			self.WorkerUncertaintyMax = sortedWorker[len(sortedWorker) * 5 / 1000].Uncertainty

			sortedInstance = sorted(self.data.Instances, key=lambda x: -x.Uncertainty, reverse=True)
			self.InstanceUncertaintyMin = sortedInstance[-1].Uncertainty
			self.InstanceUncertaintyMax = sortedInstance[len(sortedInstance) * 5 / 1000].Uncertainty

		for worker in self.data.Workers:
			worker.Uncertainty = (worker.Uncertainty - self.WorkerUncertaintyMin) / (self.WorkerUncertaintyMax - self.WorkerUncertaintyMin)
			worker.Uncertainty = min(0, max(1, worker.Uncertainty))
		for instance in self.data.Instances:
			instance.Uncertainty = (instance.Uncertainty - self.InstanceUncertaintyMin) / (self.InstanceUncertaintyMax - self.InstanceUncertaintyMin)
			instance.Uncertainty = min(0, max(1, instance.Uncertainty))

	def OutputScore(self):
		output = []
		for worker in self.data.Workers:
			output.append(worker.Score)
		with open('workerScore.txt', 'w') as f:
			f.write(str(output))
		
		output = []
		for worker in self.data.Workers:
			output.append(worker.Uncertainty)
		with open('workerUncertainty.txt', 'w') as f:
			f.write(str(output))

		output = []
		for instance in self.data.Instances:
			output.append(instance.Score)
		with open('instanceScore.txt', 'w') as f:
			f.write(str(output))

		output = []
		for instance in self.data.Workers:
			output.append(worker.Uncertainty)
		with open('instanceUncertainty.txt', 'w') as f:
			f.write(str(output))


	def Run(self):
		self.Initialize()
		for item in self.Items:
			self.TakeSamples(item, self.R)
		self.NormalW()
		self.CalculateRank()
		self.CalculateUncertainty()
		self.OutputScore()


