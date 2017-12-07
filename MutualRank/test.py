from Models import ProbabilitySparseMatrix

class test:
	def __init__(self):
		self.a =3
		return None

	def init(self):
		self.a = 1
		self.__t()

	def __t(self):
		self.a = 2

a = ProbabilitySparseMatrix(2, 3)
print (a.Srcs)
print a.SrcCount