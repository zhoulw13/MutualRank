import numpy as np

class ProbabilitySparseMatrix:
	def __init__(self, rowCount, columnCount):
		self.SrcSumValue = np.zeros(rowCount, dtype=np.float)
		self.Srcs = [np.array([], dtype=np.object) for i in range(rowCount)]
		self.SrcLastInsert = np.array([None]*rowCount)
		#self.Dests = [np.array([], dtype=np.object) for i in range(columnCount)]
		self.ForSample = np.empty(rowCount, dtype=np.object)

	def AddItem(self, src, dest, value):
		srcId = src.Index
		destId = dest.Index

		self.SrcSumValue[srcId] += value

		item = {'Src': src, 'Dest': dest, 'Value': value}
		pos = GetSrcInsertPos(self.Srcs[srcId], item, self.SrcLastInsert[srcId])
		if pos != 0:
			self.Srcs[srcId] = np.insert(self.Srcs[srcId], pos, item)
		else:
			self.Srcs[srcId] = np.append(self.Srcs[srcId], item)
		self.SrcLastInsert[srcId] = pos

	def GetSrcInsertPos(self, src, item, startpos):
		var pos = 0
		if startpos != None and src[startpos]['Dest'].Index < item['Dest'].Index:
			pos = startpos
		while src[pos] != None:
			if src[pos]['Dest'].Index > item['Dest'].Index:
				return pos
			pos += 1
		return 0

	def GetSumValue(self):
		for i in range(self.SrcCount):
			src = self.Srcs[i]
			c = len(src)
			sums = np.zeros(c, dtype=np.float)
			items = np.empty(c, dtype=np.object)

			sum = 0
			pos = 0
			while src[pos] != None:
				sum += src[pos]['Value']
				sums[pos] = sum
				items[pos] = src[pos]['Dest']
				pos += 1

			self.ForSample[i] = (sums, items)


	@property
	def SrcCount(self):
		return len(self.Srcs)


class Path:
	def __init__(self, samples):
		self.Samples = samples
		self.Alpha = np.ones(len(samples), dtype=np.double)
