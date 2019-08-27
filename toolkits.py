import numpy as np 
import math
import copy
import time

class FEmodel(object):
	"""docstring for FEmodel"""
	def __init__(self, numelem, effdof, numeig, modedata, eigvaluedata, weight):
		self.numelem = numelem
		self.effdof = effdof
		self.numeig = numeig
		self.modedata = modedata
		self.eigvaluedata = eigvaluedata
		self.weight = weight
		print('Creating a model...')

class Storey(FEmodel):
	"""docstring for Storey"""
	def __init__(self, numelem, effdof, numeig, modedata, eigvaluedata, weight, mass, kstiff):
		super(Storey, self).__init__(numelem, effdof, numeig, modedata, eigvaluedata, weight)
		self.mass = mass
		self.kstiff = kstiff
		self.dof = self.numelem
		self.stiff = np.ones((self.dof,1))
		self.stiff0 = np.ones((self.dof,1))
		self.M = np.eye(self.dof)
		self.neff = 1
		self.unknowdof = list(set(range(1,self.numelem+1))-set(self.effdof))

		_Tmat= np.eye(self.dof)
		for i in range(self.dof-1):
			_Tmat[i+1,i] = -1

		self.A = np.linalg.inv(np.transpose(_Tmat))
		self.R = np.dot(np.diagflat(np.dot(self.kstiff, self.stiff0)),_Tmat)
		print('Creating a {} storey structure...'.format(self.dof))

class Beam(FEmodel):
	"""docstring for Beam"""
	def __init__(self, numelem, effdof, numeig, modedata, eigvaluedata, weight, Toll, Dir, rhoA, kstiff):
		super(Beam, self).__init__(numelem, effdof, numeig, modedata, eigvaluedata, weight)
		self.Toll = Toll
		self.Dir = Dir
		self.rhoA = rhoA
		self.kstiff = kstiff
		self.dof = self.numelem*2
		self.neff = 2
		self.toldof = (self.numelem+1)*2
		self.stiff = np.ones((self.numelem,1))
		self.stiff0 = np.ones((self.numelem,1))
		self.unknowdof = list(set(range(1,self.numelem*2+1))-set(self.effdof))

		self.node = np.zeros(self.numelem + 1)
		self.element = np.zeros((self.numelem, 2))
		for i in range(self.numelem):
			self.node[i + 1] = Toll/self.numelem*(i + 1)
			self.element[i, 0] = i
			self.element[i, 1] = i + 1
		self.M = np.zeros((self.toldof,self.toldof))
		self.L = np.zeros((self.toldof,self.numelem*2))
		self.R = np.zeros((self.numelem*2,self.toldof))
		_s3 = math.sqrt(3)
		for iel in range(self.numelem):
			self.len = self.node[iel + 1] - self.node[iel]
			# Kiel=EI/self.len**3*np.array([[12, 6*self.len, -12, 6*self.len],[6*self.len, 4*self.len**2, -6*self.len, 2*self.len**2],[-12, -6*self.len, 12, -6*self.len],[6*self.len, 2*self.len**2, -6*self.len, 4*self.len**2]])  
			
			Miel_=self.rhoA*self.len/420*np.array([[156, 22*self.len, 54, -13*self.len],
				  [22*self.len, 4*self.len**2, 13*self.len, -3*self.len**2],
				  [54, 13*self.len, 156, -22*self.len],
				  [-13*self.len, -3*self.len**2, -22*self.len, 4*self.len**2]]) 
			LA=np.array([[0, 2*_s3],[self.len, _s3*self.len],
				[0, -2*_s3],[-self.len, _s3*self.len]])

			# K[iel: iel + 4, iel: iel + 4]= K(iel: iel + 4,iel: iel + 4) + stiff(iel)*Kiel
			self.M[iel*2: iel*2 + 4, iel*2: iel*2 + 4]= self.M[iel*2: iel*2 + 4, iel*2:iel*2 + 4] + Miel_
			self.L[iel*2: iel*2 + 4, iel*2: iel*2 + 2]= self.L[iel*2: iel*2 + 4, iel*2: iel*2 + 2] + LA
			self.R[iel*2: iel*2 + 2, iel*2: iel*2 + 4]= self.R[iel*2: iel*2 + 2, iel*2: iel*2 + 4] + self.kstiff[iel]/self.len**3*LA.T
		dofs_with_constrain = list(set(range(1, self.toldof + 1)) - set(Dir))
		dofs_with_constrain = [x-1 for x in dofs_with_constrain]
		_Mt = self.M[dofs_with_constrain, :]
		self.M = _Mt[:, dofs_with_constrain]
		self.L = self.L[dofs_with_constrain, :]
		self.R = self.R[:, dofs_with_constrain]


		U,S,V =np.linalg.svd(self.L)
		V = V.T
		self.A=np.dot(np.dot(V,np.diag(1/S)),U.T)

		print('Creating a {} element beam structure...'.format(self.numelem))



class Optimizer(object):
	"""docstring for Optimizer"""
	def __init__(self, goal, tol, nmax):
		self.goal = goal
		self.tol = tol
		self.nmax = nmax
			
class DecoupledOptimizer(Optimizer):
	"""docstring for DecoupledOptimizer"""
	def __init__(self, goal, tol, nmax):
		super(DecoupledOptimizer, self).__init__(goal, tol, nmax)
		self.mu = None #??
		print('Optimizer: DecoupledOptimizer.')

	def optimize(self):
		print('Optimizing...\n')
		count_ = 0
		change_ = 1
		stiffrecord_ = self.goal.model.stiff0
		while change_>self.tol and count_<self.nmax:
			count_ += 1
			a_ = copy.deepcopy(self.goal.a)
			b_ = copy.deepcopy(self.goal.b)
			if self.goal.regflag:
				self.decide_reg_param(a_, b_)
				for j in range(self.goal.model.numelem):
					x_ = self.goal.model.stiff0[j,0]
					if self.mu <= 2*(a_[j,0]*x_ - b_[j,0]):
						self.goal.model.stiff[j,0] = (b_[j,0] + 0.5*self.mu)/a_[j,0]
					elif self.mu <= 2*(-a_[j,0]*x_ + b_[j,0]):
						self.goal.model.stiff[j,0] = (b_[j,0] - 0.5*self.mu)/a_[j,0]
					else:
						self.goal.model.stiff[j,0] = x_
			else:
				self.goal.model.stiff = np.sqrt(self.goal.b/self.goal.a)
			dstiff_ = self.goal.model.stiff - stiffrecord_
			stiffrecord_ = copy.deepcopy(self.goal.model.stiff)
			change_ = np.linalg.norm(dstiff_)/np.linalg.norm(self.goal.model.stiff0)

		return self.goal.model.stiff

	def decide_reg_param(self, a, b):
		lmax_ = np.min([self.goal.lmax, self.goal.model.numelem])
		mucr_ = 2*np.abs(b - a*self.goal.model.stiff0)
		mucrd_ = sorted(mucr_[:,0], reverse=True)
		j_ = 0
		er1_ = mucrd_[0] - self.goal.alpha*mucrd_[1]
		temp_ = lmax_ - 2
		while j_ < temp_ and er1_ < 0:
			er1_ = mucrd_[j_] - self.goal.alpha*mucrd_[j_+1]
			j_ = j_ + 1
		self.mu = mucrd_[j_+1]
		return self.mu	

class Goal(object):
	"""docstring for Goal"""
	def __init__(self, model):
		self.model = model
		self.regflag = False
		self.lmax = None
		self.alpha = None

	def using_sparse_reg(self, lmax, alpha):
		self.regflag = True
		self.lmax = lmax
		self.alpha = alpha
		print('Using sparse regularation with L-max={}, \u03B1={}...'.format(lmax, alpha))

class DecoupledGoal(Goal):
	"""docstring for DecoupledGoal"""
	def __init__(self, model):
		super(DecoupledGoal, self).__init__(model)
		# self.stiff = self.model.stiff
		self._completemode = None
		self._a = None
		self._b = None	
		print('Creating the decoupled goal function...')

	@property
	def completemode(self):
		self._completemode = np.zeros((self.model.dof, self.model.numeig))
		self._completemode[[x-1 for x in self.model.effdof],:] = self.model.modedata
		_KD = np.zeros((self.model.dof, self.model.dof))
		if self.model.neff ==1:
			for i in range(self.model.numelem):
				_KD[i,i] = self.model.stiff[i,0]
		else:
			for i in range(self.model.numelem):
				_KD[2*i,2*i] = self.model.stiff[i,0]
				_KD[2*i+1,2*i+1] = self.model.stiff[i,0]
		for i in range(self.model.numeig):
			_AuxMat = np.dot(_KD, self.model.R) - np.dot(self.model.A, self.model.M)*self.model.eigvaluedata[i]
			_AA = np.dot(np.transpose(_AuxMat), _AuxMat)
			_AAtemp = _AA[[x-1 for x in self.model.unknowdof], :]
			_AAtemp1 = _AAtemp[:, [x-1 for x in self.model.unknowdof]]
			_AAtemp2 = _AAtemp[:, [x-1 for x in self.model.effdof]]
			self._completemode[[x-1 for x in self.model.unknowdof], i] = -np.dot(np.linalg.inv(_AAtemp1), 
											 np.dot(_AAtemp2.reshape(len(self.model.unknowdof),len(self.model.effdof)),self.model.modedata[:,i]))
		return self._completemode

	@property
	def a(self):
		_lamda = np.zeros((self.model.numeig, self.model.numeig))
		_completemode = copy.deepcopy(self.completemode)
		for i in range(self.model.numeig):
			_lamda[i,i] = self.model.eigvaluedata[i]


		_vecta = np.dot(self.model.R, _completemode)
		_vecta2 = np.dot(_vecta*_vecta, self.model.weight)
		if self.model.neff==1:
			self._a = np.zeros((self.model.dof, 1))
			for i in range(self.model.dof):
				self._a[i,0] = sum(_vecta2[i,:])
		else:
			_at = np.zeros((self.model.dof, 1))
			for i in range(self.model.dof):
				_at[i,0] = sum(_vecta2[i,:])	
			self._a = np.zeros((self.model.numelem, 1))
			for i in range(self.model.numelem):
				self._a[i,0] = _at[2*i,0] + _at[2*i + 1,0] 
		return self._a

	@property
	def b(self):
		_lamda = np.zeros((self.model.numeig, self.model.numeig))
		_completemode = copy.deepcopy(self.completemode)
		for i in range(self.model.numeig):
			_lamda[i,i] = self.model.eigvaluedata[i]

		_vecta = np.dot(self.model.R, _completemode)
		_vectb = np.dot(np.dot(self.model.A, self.model.M), np.dot(_completemode, _lamda))
		_vectba = np.dot(_vectb*_vecta, self.model.weight)
		if self.model.neff==1:	
			self._b = np.zeros((self.model.dof, 1))
			for i in range(self.model.dof):
				self._b[i,0] = sum(_vectba[i,:])
		else:
			_bt = np.zeros((self.model.dof, 1))
			for i in range(self.model.dof):
				_bt[i,0] = sum(_vectba[i,:])	
			self._b = np.zeros((self.model.numelem, 1))
			for i in range(self.model.numelem):
				self._b[i,0] = _bt[2*i,0] + _bt[2*i + 1,0]				
		return self._b

	def optimize(self, optimizer=DecoupledOptimizer, tol=1e-7, nmax=1000):
		print('Ready for optimization with:')
		self.results = optimizer(self,tol,nmax).optimize()
		return self.results


if __name__ == '__main__':
	time_start=time.time()
	numelem = 20     #number of elements
	MeaNodes = [20]
	effdof = [2*MeaNode-1  for MeaNode in MeaNodes]
	numeig = 6

	OrderUse = [i for i in range(numeig)]

	eigenvalue_data= 2*math.pi*np.array([23.09, 140.90, 407.75, 795.14, 1292.30, 1998.43])
	eigenvalue_data= eigenvalue_data*eigenvalue_data
	mode_data = np.ones((1, numelem))
	eigvaluedata = eigenvalue_data[OrderUse]
	modedata = mode_data[:, OrderUse]
	weight = np.eye(numeig)

	Toll = 0.4953
	Em=7.1e10
	rhom=2.21e3
	Im=1/12*0.0254*0.00635**3
	Am=0.0254*0.00635
	Dir=[1, 2]
	rhoA=rhom*Am
	kstiff = Em*Im*np.ones(numelem)


	print(numelem,effdof,numeig,modedata,eigvaluedata,weight,Toll,Dir,rhoA,kstiff)
	# Generate beam model with both FEM model & actual data.
	beam20 = Beam(numelem=numelem, effdof=effdof, numeig=numeig, modedata=modedata,
					 eigvaluedata=eigvaluedata, weight=weight, Toll=Toll, Dir=Dir, rhoA=rhoA, kstiff=kstiff)
	# Generate goal function.
	goal = DecoupledGoal(beam20)
	# Adding sparse regularation.
	goal.using_sparse_reg(lmax=4, alpha=100)
	# Optimize the goal function.
	goal.optimize(tol=1e-6,nmax=1000)
	# Print the optimization results.
	print('The optimized stiff:\n {}\n'.format(goal.results))

	time_end=time.time()
	print('totally cost',time_end-time_start)
