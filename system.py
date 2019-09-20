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
import random

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
	ax.plot(Frequencies,s1)
	ax.plot(Frequencies,s2)
	ax.plot(Frequencies,s3)
	ax.set_xlim([0,30])

@eel.expose
def creatStoreyModeShapeJSCAD(mode,order,dof):
	plate = ''
	plate += "function getParameterDefinitions(){return [{ name: 'numMode', caption: '模态阶数:', type: 'int', initial: 1, min: 1, max: "
	plate += str(order) 
	plate += ", step: 1 }];}function main(params) {var storey = involuteStorey(params.numMode);return storey;}function involuteStorey(numMode) {var storeyMode = ["
	for o in range(order):
		plate += "["
		for d in range(dof):
			plate += "{'dof':" + str(d+1) + ",'value':" + str(mode[o*dof+d]) + "},"
		plate += "],"
	plate += "];var numStorey = "
	plate += str(dof)
	plate += ";var pillarHeight = 10;var storeyWidth = 10;var pillarPitch = 0.5;numMode = numMode - 1;var ratio = 0.5;var group = [];for (var k = 0; k < numStorey; k++) {if(k < 1){var pillar1 = CSG.cylinder({start: [-storeyWidth/2, -storeyWidth/2, k*1.1*pillarHeight],end: [-storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});var pillar2 = CSG.cylinder({start: [-storeyWidth/2, storeyWidth/2, k*1.1*pillarHeight], end: [-storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});var pillar3 = CSG.cylinder({start: [ storeyWidth/2, -storeyWidth/2, k*1.1*pillarHeight],end: [ storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});var pillar4 = CSG.cylinder({start: [ storeyWidth/2, storeyWidth/2, k*1.1*pillarHeight],end: [ storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});}else{pillar1 = CSG.cylinder({start: [-storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k - 1].value - (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05],end: [-storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value + (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});pillar2 = CSG.cylinder({start: [-storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k - 1].value - (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05],end: [-storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value + (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});pillar3 = CSG.cylinder({start: [ storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k - 1].value - (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05],end: [ storeyWidth/2, -storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value + (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});pillar4 = CSG.cylinder({start: [ storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k - 1].value - (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05],end: [ storeyWidth/2, storeyWidth/2 + storeyWidth*ratio*storeyMode[numMode][k].value + (storeyWidth*ratio*storeyMode[numMode][k].value - storeyWidth*ratio*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05],radius: pillarPitch, resolution: 200});}var storey = cube({size: [storeyWidth*1.2, storeyWidth*1.2, pillarHeight*0.1],center: [true,true,false]}).translate([0,storeyWidth*ratio*storeyMode[numMode][k].value,pillarHeight+(pillarHeight*1.1)*k]);group.push(pillar1,pillar2,pillar3,pillar4,storey)}return group}"
	
	
	try:
		with open("./web/models/storeyModeShape.jscad",'w',encoding="utf-8") as f:
			f.write(plate)
		return 1
	except Exception as e:
		return 2


@eel.expose
def FDD(csvfile,fs,order,nperseg,window):
	print('Here python!')
	# print(type(csvfile))
	fs = int(fs)
	nperseg = int(nperseg)
	# print(csvfile,fs,nperseg,window)
	# print(type(csvfile),type(fs),type(nperseg),type(window))
	data = pd.read_csv(csvfile,header=None)

	if window == '汉宁窗':
		window = 'hann'
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
	s1 = np.zeros(PSD.shape[2])
	for i in range(PSD.shape[2]):
		u, s, _ = np.linalg.svd(PSD[:,:,i])
		s1[i] = s[0]
		ms[:,i] = u[:,0]
	s1=20*np.log10(s1)

	peaks, _= signal.find_peaks(s1, height=10)
	omega = Frequencies[peaks][0:order]
	Phi = ms[:,peaks][:,0:order]
	Phi_amp = np.abs(Phi).T
	Phi_list = Phi_amp.reshape(col*order).tolist()

	results = []
	results.append(Phi_list)
	results.append(omega.tolist())
	results.append(col)
	return results

@eel.expose
def SSI(csvfile,fs,order,s):
	print("Here python, SSI!")
	fs = int(fs)
	orders = order
	order = 2*int(order)
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
	Phi = np.dot(C,phi)
	Phi_amp = np.abs(Phi).T
	Phi_amps = np.zeros((orders,ns))
	for i in range(orders):
		Phi_amps[i, :] = Phi_amp[2*i, :]
	Phi_list = Phi_amps.reshape(ns*orders).tolist()
	results = []
	results.append(Phi_list)
	results.append(omega)
	results.append(ns)
	return results

def on_close(page, sockets):
	print(page, 'closed')
	print('Still have sockets open to', sockets)

@eel.expose
def StoreyDetect(goaltype,csvfile,dof,effdof,orderuse,lmax,alpha,tol,nmax,reg=True):
	# goaltype: string
	# csvfile: string(dir)
	# dof: integer
	# effdof: list
	# orderuse: list
	# print('Here python!')
	# print(type(goaltype), type(csvfile), type(dof), type(effdof), type(orderuse))
	# print(goaltype, csvfile, dof, effdof, orderuse)


	neffdof = len(effdof)
	numeig = len(orderuse) 
	data_ = pd.read_csv(csvfile, header=None).values
	kstiff = data_[:,0]
	kstiff = np.diagflat(kstiff)
	mass = data_[:,1]
	if neffdof == 1:
		modedata = data_[:,2:2+neffdof]
		modedata = modedata[[x-1 for x in orderuse],:].reshape((numeig, neffdof)).T
		eigvaluedata0 = data_[:,2+neffdof]
		eigvaluedata1 = data_[:,2+neffdof+1]
		eigvaluedatafem = data_[:,-1]
	else:
		modedata0 = data_[[x-1 for x in orderuse]][:,2:2+neffdof]
		modedata1 = data_[[x-1 for x in orderuse]][:,2+neffdof:2+neffdof*2]
		modefem = data_[[x-1 for x in orderuse]][:,2+neffdof*2:2+neffdof*3]
		modefem = modefem/modefem[-1,-1]
		modedata = modefem + (modedata1 - modedata0)
		modedata = modedata.reshape((numeig, neffdof)).T
		eigvaluedata0 = data_[[x-1 for x in orderuse]][:,2+neffdof*3]
		eigvaluedata1 = data_[[x-1 for x in orderuse]][:,2+neffdof*3+1]
		eigvaluedatafem = data_[[x-1 for x in orderuse]][:,-1]

	eigvaluedata0 = 2*math.pi*eigvaluedata0
	eigvaluedata1 = 2*math.pi*eigvaluedata1
	eigvaluedata0 = eigvaluedata0*eigvaluedata0
	eigvaluedata1 = eigvaluedata1*eigvaluedata1
	eigvaluedata = eigvaluedatafem + eigvaluedatafem/eigvaluedata0*(eigvaluedata1-eigvaluedata0)
	eigvaluedata = eigvaluedata[[x-1 for x in orderuse]]
	weight = np.diag(1/eigvaluedata)

	# Generate model.
	model_ = det.Storey(numelem=dof, effdof=effdof, numeig=numeig, modedata=modedata,
						eigvaluedata=eigvaluedata, weight=weight, mass=mass, kstiff=kstiff)
	# Generate goal function.
	if goaltype == "解耦型":
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
def BeamDetect(goaltype,csvfile,numelem,MeasuredNodes,orderuse,DirDOF,TolLen,E,rho,area,Im,lmax,alpha,tol,nmax,reg=True):
	# goaltype: string
	# csvfile: string(dir)
	# numelem: integer
	# MeasuredNodes: list
	# orderuse: list
	# DirDOF: list
	# TolLen: float
	# E: float
	# rho: float
	# area: float
	# Im: float
	# print('Here python!')
	# print(type(goaltype), type(csvfile), type(numelem), type(MeasuredNodes), type(orderuse))
	# print(goaltype, csvfile, numelem, MeasuredNodes, orderuse)

	data_ = pd.read_csv(csvfile, header=None).values
	nmeanodes = len(MeasuredNodes)
	effdof = [2*MeasuredNode-1  for MeasuredNode in MeasuredNodes]
	orderuse = [i-1 for i in orderuse]
	numeig = len(orderuse)
	mode_data = data_[:,0:nmeanodes].reshape((numeig, nmeanodes)).T
	eigenvalue_data= data_[:,nmeanodes].reshape(numeig)
	eigenvalue_data= eigenvalue_data*2*np.pi
	eigenvalue_data= eigenvalue_data*eigenvalue_data
	eigvaluedata = eigenvalue_data[orderuse]
	modedata = mode_data[:, orderuse]
	weight = np.eye(numeig)
	rhoA=rho*area
	kstiff = E*Im*np.ones(numelem)
	
	# print(numelem,effdof,numeig,modedata,eigvaluedata,weight,TolLen,DirDOF,rhoA,kstiff)

	# Generate model.
	model_ = det.Beam(numelem=numelem, effdof=effdof, numeig=numeig, modedata=modedata,
					  eigvaluedata=eigvaluedata, weight=weight, Toll=TolLen, Dir=DirDOF, rhoA=rhoA, kstiff=kstiff)
	# Generate goal function.
	if goaltype == "解耦型":
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
    }
	eel.init('web')
	eel.start('index.html?v={:.8f}'.format(random.random()), size=(1920,1080), options=my_options, callback=on_close)


	