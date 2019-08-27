import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import eel
import math
import toolkits as det
from scipy import signal 
from tkinter import filedialog
from tkinter import *
from mpld3 import plugins
from sklearn.utils.extmath import randomized_svd


@eel.expose
def getPath():
	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)
	filepath =  filedialog.askopenfilename(initialdir = "./",title = "Select data file",filetypes = (("csv files","*.csv"),("all files","*.*")))
	return filepath

def Identifier(PSD,Frequencies,ax):
	s1 = []
	s2 = []
	s3 = []
	for i in range(PSD.shape[2]):
		u, s, _ = np.linalg.svd(PSD[:,:,i])
		s1.append(s[0])
		s2.append(s[1])
		s3.append(s[2])
		ms[:,i] = u[:,0]
	# ax.plot(Frequencies,20*np.log10(s1))
	# ax.plot(Frequencies,20*np.log10(s2))
	# ax.plot(Frequencies,20*np.log10(s3))
	ax.plot(Frequencies,s1)
	ax.plot(Frequencies,s2)
	ax.plot(Frequencies,s3)
	ax.set_xlim([0,30])


@eel.expose
def FDD(csvfile,fs,nperseg,window):
	# data = pd.read_csv('Accelerations.csv',header=None)
	print('Here python!')
	# print(type(csvfile))
	fs = int(fs)
	nperseg = int(nperseg)
	print(csvfile,fs,nperseg,window)
	print(type(csvfile),type(fs),type(nperseg),type(window))
	data = pd.read_csv(csvfile,header=None)

	test, _= signal.csd(data[0],data[0],fs=fs,window=window,nperseg=nperseg,detrend=False)

	(row,col) = data.shape
	fwidth = test.shape[0]

	PSD = np.zeros((col,col,fwidth),dtype=np.complex_)
	F = np.zeros((col,col,fwidth))
	ms = np.zeros((col,row),dtype=np.complex_)

	for i in range(col):
		for j in range(col):
			F[i][j], PSD[i][j]= signal.csd(data[i],data[j],fs=fs,window=window,nperseg=nperseg,detrend=False)

	Frequencies = F[1,1,:]
	fig, ax = plt.subplots(1, 1)
	fig.suptitle('figure') 
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('1st Singular values of the PSD matrix (db)')
	s1 = []
	s2 = []
	s3 = []
	for i in range(PSD.shape[2]):
		u, s, _ = np.linalg.svd(PSD[:,:,i])
		s1.append(s[0])
		s2.append(s[1])
		s3.append(s[2])
		ms[:,i] = u[:,0]
	# ax.plot(Frequencies,20*np.log10(s1))
	# ax.plot(Frequencies,20*np.log10(s2))
	# ax.plot(Frequencies,20*np.log10(s3))
	ax.plot(Frequencies,s1)
	ax.plot(Frequencies,s2)
	ax.plot(Frequencies,s3)
	peaks, _ = signal.find_peaks(s1)
	s1 = np.array(s1)
	ax.plot(Frequencies[peaks],s1[peaks],"o")
	plugins.connect(fig, plugins.MousePosition(fontsize=14))
	ax.grid(True, alpha=0.3)
	ax.set_xlim([0,30])

	# encoded = fig_to_base64(fig)
	# figure = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
	# return figure
	head = '<!DOCTYPE html><html><head><title></title></head><body>'
	tail = '</body></html>'
	html = head + mpld3.fig_to_html(fig) + tail
	with open('./web/fddfigure.html','w') as f:
		f.write(html)
	return True

@eel.expose
def SSI(csvfile,fs,order,s):
	print("Here python, SSI!")
	fs = int(fs)
	order = int(order)
	s = int(s)
	Y = pd.read_csv(csvfile, header=None)
	Y = Y.values
	nt=np.size(Y,0)
	ns=np.size(Y,1)
	nb=2*s

	Rxy=np.zeros((nb,ns,ns))
	for i in range(ns):
		for j in range(ns):
			rxy = signal.correlate(Y[:,i],Y[:,j])
			for k in range(nb):
				Rxy[k,j,i]=rxy[nt-1-k]/(nt-k)

	Hank=np.zeros((s*ns,s*ns))
	for i in range(s):
		for j in range(s):
			Hank[i*ns:(i+1)*ns,j*ns:(j+1)*ns] = Rxy[i+j+1,:,:]

	del Rxy
	U1, S, _ = randomized_svd(Hank, n_components=order, n_iter=7, random_state=42)
	del Hank
        
	S1=np.diag(S)

	gam  = np.dot(U1,np.sqrt(S1))
	syt1=[j for j in range(ns*(s-1))]
	gamm = np.dot(U1[syt1,:],np.sqrt(S1))
	gamm_inv = np.linalg.pinv(gamm)
	syt2=[j for j in range(ns,ns*s)]
	A = np.dot(gamm_inv,gam[syt2,:])
	syt3=[j for j in range(ns)]
	C = gam[syt3,:]

	lamda,phi=np.linalg.eig(A)
	omega=np.zeros(order)
	for i in range(order):
	    omega[i]=np.abs(fs*np.log(lamda[i]))/2/np.pi

	omega=list(sorted(set(omega)))
	Phi=np.dot(C,phi)
	Phi = Phi.tolist()
	# print(omega,Phi)
	# print(type(omega),type(Phi))
	return omega

def on_close(page, sockets):
	print(page, 'closed')
	print('Still have sockets open to', sockets)

@eel.expose
def Detect(modeltype,goaltype,csvfile,dof,effdof,orderuse,reg=True,lmax=3,alpha=10,tol=1e-6,nmax=1000):
	# modeltype: string
	# goaltype: string
	# csvfile: string(dir)
	# dof: integer
	# effdof: list
	# orderuse: list
	print('Here python!')
	print(type(modeltype),type(goaltype),type(csvfile),type(dof),type(effdof),type(orderuse))
	data_ = pd.read_csv(csvfile, header=None).values
	kstiff = data_[:,0]
	kstiff = np.diagflat(kstiff)
	mass = data_[:,1]
	modedata = data_[:,2:2+len(effdof)]
	modedata = modedata[[x-1 for x in orderuse],:].reshape((1,len(orderuse)))
	eigvaluedata0 = data_[:,2+len(effdof)]
	eigvaluedata1 = data_[:,2+len(effdof)+1]
	eigvaluedata0 = 2*math.pi*eigvaluedata0
	eigvaluedata1 = 2*math.pi*eigvaluedata1
	eigvaluedata0 = eigvaluedata0*eigvaluedata0
	eigvaluedata1 = eigvaluedata1*eigvaluedata1
	eigvaluedatafem = data_[:,-1]
	eigvaluedata = eigvaluedatafem + eigvaluedatafem/eigvaluedata0*(eigvaluedata1-eigvaluedata0)
	eigvaluedata = eigvaluedata[[x-1 for x in orderuse]]
	weight = np.diag(1/eigvaluedata)
	print(dof)
	print(effdof)
	print(len(orderuse))
	print(type(modedata),modedata.shape)
	print(modedata)
	print(eigvaluedata)
	print(type(kstiff),kstiff.shape)
	print(kstiff)

	# Generate model.
	if modeltype == "Storey":
		model_ = det.Storey(numelem=dof, effdof=effdof, numeig=len(orderuse), modedata=modedata,
					 eigvaluedata=eigvaluedata, weight=weight, mass=mass, kstiff=kstiff)
	else:
		pass
	# Generate goal function.
	if goaltype == "Decoupled":
		goal_ = det.DecoupledGoal(model_)
	else:
		pass
	# Adding sparse regularation.
	if reg:
		goal_.using_sparse_reg(lmax=lmax, alpha=alpha)
	# Optimize the goal function.
	goal_.optimize(tol=tol,nmax=nmax)
	# Print the optimization results.
	print('The optimized stiff:\n {}\n'.format(goal_.results))

	results = []

	for i in range(dof):
		resultsdict = {}
		resultsdict['dof'] = str(i+1)
		resultsdict['value'] = goal_.results[i,0]
		results.append(resultsdict)

	return results

@eel.expose
def BeamDetect(modeltype,goaltype,csvfile,numelem,MeasuredNodes,orderuse,DirDOF,TolLen,E,rho,area,Im,reg=True,lmax=4,alpha=100,tol=1e-6,nmax=1000):
	# modeltype: string
	# goaltype: string
	# csvfile: string(dir)
	# numelem: integer
	# MeasuredNodes: list
	# orderuse: list
	# E: float
	# rho: float
	# area: float
	# Im: float
	print('Here python!')
	print(type(modeltype),type(goaltype),type(csvfile),type(numelem),type(MeasuredNodes),type(orderuse))
	data_ = pd.read_csv(csvfile, header=None).values
	# kstiff = data_[:,0].reshape((dof,1))
	# mass = data_[:,1]
	# modedata = data_[:,2:2+len(effdof)]
	# modedata = modedata[[x-1 for x in orderuse],:].reshape((len(orderuse),))
	# eigvaluedata0 = data_[:,2+len(effdof)]
	# eigvaluedata1 = data_[:,2+len(effdof)+1]
	# eigvaluedata0 = 2*math.pi*eigvaluedata0
	# eigvaluedata1 = 2*math.pi*eigvaluedata1
	# eigvaluedata0 = eigvaluedata0*eigvaluedata0
	# eigvaluedata1 = eigvaluedata1*eigvaluedata1
	# eigvaluedatafem = data_[:,-1]
	# eigvaluedata = eigvaluedatafem + eigvaluedatafem/eigvaluedata0*(eigvaluedata1-eigvaluedata0)
	# eigvaluedata = eigvaluedata[[x-1 for x in orderuse]]
	# print(dof)
	# print(effdof)
	# print(len(orderuse))
	# print(type(modedata),modedata.shape)
	# print(eigvaluedata)
	# print(type(kstiff),kstiff.shape)
	
	effdof = [2*MeasuredNode-1  for MeasuredNode in MeasuredNodes]
	orderuse = [i-1 for i in orderuse]
	numeig = len(orderuse)

	# mode_data = np.ones((1, numelem))
	mode_data = data_[:,0].reshape((1,numeig))
	# eigenvalue_data= 2*math.pi*np.array([23.09, 140.90, 407.75, 795.14, 1292.30, 1998.43])
	eigenvalue_data= data_[:,1].reshape(numeig)
	eigenvalue_data= eigenvalue_data*2*np.pi
	eigenvalue_data= eigenvalue_data*eigenvalue_data
	eigvaluedata = eigenvalue_data[orderuse]
	modedata = mode_data[:, orderuse]
	

	weight = np.eye(numeig)
	
	E=7.1e10
	rho=2.21e3
	area=0.0254*0.00635
	Im=1/12*0.0254*0.00635**3
	
	rhoA=rho*area
	kstiff = E*Im*np.ones(numelem)
	
	print(numelem,effdof,numeig,modedata,eigvaluedata,weight,TolLen,DirDOF,rhoA,kstiff)

	# Generate model.
	if modeltype == "Beam":
		model_ = det.Beam(numelem=numelem, effdof=effdof, numeig=numeig, modedata=modedata,
					 eigvaluedata=eigvaluedata, weight=weight, Toll=TolLen, Dir=DirDOF, rhoA=rhoA, kstiff=kstiff)
	else:
		pass
	# Generate goal function.
	if goaltype == "Decoupled":
		goal_ = det.DecoupledGoal(model_)
	else:
		pass
	# Adding sparse regularation.
	if reg:
		goal_.using_sparse_reg(lmax=lmax, alpha=alpha)
	# Optimize the goal function.
	goal_.optimize(tol=tol,nmax=nmax)
	# Print the optimization results.
	print('The optimized stiff:\n {}\n'.format(goal_.results))

	results = []

	for i in range(numelem):
		resultsdict = {}
		resultsdict['dof'] = str(i+1)
		resultsdict['value'] = goal_.results[i,0]
		results.append(resultsdict)

	return results

if __name__ == '__main__':
	my_options = {
    'mode': "chrome-app", #or "chrome-app",
    'host': 'localhost',
    'port': 8080,
    # 'chromeFlags': ["--start-fullscreen", "--browser-startup-dialog"]
    }
	# eel.start('base1.html', options=my_options)
	eel.init('web')
	# eel.start('base.html', size=(1200,800), options=my_options, callback=on_close)
	eel.start('index.html', size=(1200,900), options=my_options, callback=on_close)


	