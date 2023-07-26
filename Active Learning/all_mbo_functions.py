import random
import os, sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Fragments as Fragments
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Lipinski as Lipinski
from rdkit.Chem import rdMolDescriptors
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.Descriptors3D as Descriptors3D

from copy import copy
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C ,WhiteKernel as Wht,Matern as matk
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


import math
import torch, gpytorch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE

def select_initial_smi(n_data,n_ini):
    """
    Randomly select initial smiles for training 
    Input: 
    ndata = total number of samples
    ntrainInit = number of points for initial training
    Output:
    Two arrays of indexes of ndata: sample_idx, and remaining_idx 
    Use: 
    test_data_idx,remaing_data_idx = get_samples(ndata,ntrainInit)
    """
    nremain = n_data - n_ini
    dataset = np.random.permutation(n_data)
    x_ini = np.empty(n_ini, dtype=int) # randomly chosen data points
    x_remain = np.empty(nremain, dtype=int) # remaining data points
    x_ini[:] = dataset[0:n_ini]
    x_remain[:] = dataset[n_ini:n_data]
    return x_ini,x_remain

def generate_initial_remain_smi(path_file_smi,n_ini):
    """
    This function takes the path of the file_smi.csv, and the number of initial smi for training.
    
    Input_smi --> Molecular descriptor generations --> Apply constraints --> Writes df_molD_smi
    
    Output: Writes two files path_file_smi_initial.csv and path_file_smi_remain.csv
    """
    file_name=os.path.splitext(os.path.basename(path_file_smi))[0] 
    print("File name: ", file_name)
    all_smi=pd.read_csv(path_file_smi)
    print("Generating Mol. descriptors")
    df_mold=generate_molDescriptors(all_smi["SMILES"])

    print("Applying constraints")
    df_mold=df_mold[df_mold["number_heavy_atoms"]-40<=0] 
    df_mold=df_mold[df_mold["SMILES"].str.count("C#N") < 2] # Maximum 1 C#N 
    df_mold=df_mold[df_mold["SMILES"].str.count("N#C") < 1] # remove remaining N#C
    df_mold=df_mold[df_mold["SMILES"].str.count("\[N\+\]\(\=O\)\[O\-\]") < 2] # Maximum 1 NO2
    print("Shape of the df_mol_descriptors after applying constrainsts = ", df_mold.shape)

    ini_smi,remain_smi = select_initial_smi(n_data=df_mold.shape[0],n_ini=int(n_ini))
    df_remain=df_mold.iloc[remain_smi]
    
    if len(ini_smi):
        df_ini=df_mold.iloc[ini_smi]

        df_ini.to_csv(file_name+"_initial.csv",index=False)
        df_remain.to_csv(file_name+"_remain.csv",index=False)
        print(file_name+"_initial.csv and "+file_name+"_remain.csv saved in: ", os.getcwd())
        return None
    else:
        print("No SMILES selected for initial training")
        df_remain.to_csv(file_name+"_remain.csv",index=False)
        print(file_name+"_remain.csv saved in: ", os.getcwd())
        return None

def do_scaling(scaler=StandardScaler(), xtrain=None, xtest=None):
    """
    Usage: do_scaling(scaler=MinMaxScaler(), xtrain=xtrain, xtest=test) 
    xtrain and xtest are pd.Dataframes
    Caution: Do test_train_split before scaling
    Return: return scaled non-None xtrain and xtest
    """
    st = scaler

    if xtrain is not None:
        col=xtrain.columns.values.tolist()
        xtrain=st.fit_transform(xtrain)  
        xtrain=pd.DataFrame(xtrain,columns=col)

        if xtest is not None:
            
            xtest=st.transform(xtest)
            xtest=pd.DataFrame(xtest,columns=col)
            print("returning scaled train and test data")
            return xtrain,xtest
        else:
            print("test data is not provided, returning only scaled train data")
            return xtrain
    else:
        print("Give train data, returning None")
        return xtrain,xtest

def do_pca(xtrain=None, xtest=None, rvar=0.95):
    """
    Usage: do_pca(xtrain=xtrain, xtest=test) 
    Caution: Do test_train_split and scaling before pca
    Return: Transformed xtrain and xtest if they are not None
    """
    
    if xtrain is not None:
        
        pca = PCA().fit(xtrain)
        evr = np.cumsum(pca.explained_variance_ratio_)
        n_comp = 1+np.nonzero(evr > rvar)[0][0]

        print(str(n_comp)+" principal components can describe > "+ str(rvar*100)+ "% of variance in the data")
        print("Selected "+str(n_comp)+" components for PCA")
        
        col=[]
        for i in range(1,1+n_comp,1):
            col.append("PC"+str(i))

        pca = PCA(n_components=n_comp)
        xtrain = pca.fit_transform(xtrain) 
        xtrain=pd.DataFrame(xtrain,columns=col)

        if xtest is not None:
            
            xtest=pca.transform(xtest)
            xtest=pd.DataFrame(xtest,columns=col)
            print("returning pca transformed train and test data")
            return xtrain,xtest
        else:
            print("test data is not provided, returning only transformed train data")
            return xtrain
    else:
        print("Give train data, returning None")
        return xtrain,xtest


def smiles2dotcom(list_of_smiles,method="wB97XD",basis="CEP-31G",Freq=" ",sol=" ",charge=0,mult=1,fileName="g16_input"):

    """ 
    This function will take a list of smiles (or a Pandas series of SMILES) and generate .com files (in current folder)
    For single SMILES string: smiles_to_dotcom(["CC"])
    """
    def get_atoms(mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        return atoms
    
    def generate_structure_from_smiles(smiles):
        # Generate a 3D structure from smiles
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol)
        status = AllChem.MMFFOptimizeMolecule(mol) #UFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)
        atoms = get_atoms(mol)
        return atoms, coordinates

    def mkGaussianInputScriptNeutral(comName,method,basis, fragment,atoms,coordinates,charge,mult):
        file = open(comName+".com", 'w')
        file.write("""%mem=16GB \n""")
        file.write("""%Chk="""+comName+""".chk \n""")
        file.write("""#p """+ method +"""/""" + basis+ " opt " + Freq + "   " + sol + """\n\n""")
        file.write(fragment + " " + str(charge) + " " + str(mult)+"\n\n")
        file.write(str(charge)+""" """+str(mult)+"""\n""")
        for i,atom in enumerate(atoms):
            file.write(str(atom) + "\t"+str(coordinates[i][0]) + "\t\t"+str(coordinates[i][1]) + "\t\t"+str(coordinates[i][2]) + "\n")
        file.write("\n")
        file.close()   

    for i,smilesName in enumerate(list_of_smiles):
        atoms,coordinates=generate_structure_from_smiles(smilesName)
        fileNameNeutral = fileName + "-" + method +"-"+str(i+1)
        mkGaussianInputScriptNeutral(fileNameNeutral,method,basis,smilesName, atoms, coordinates,charge, mult)
    
    print("Files generated in: ", os.getcwd())

def gpregression_pytorch(X_train,y_train,num_iter=200,learning_rate=0.1,verbose=False):
    
    """
    Gaussian Process Regression implementation with the GPyTorch
    
    Input: gpregression_pytorch(X_train,y_train,num_iter,learning_rate)
    
    Retrun: trained_model, trained_likelihood

    """
    nfeatures=X_train.shape[1]
    train_x=torch.from_numpy(np.array(X_train))
    train_y=torch.from_numpy(np.array(y_train))
    torch.set_default_dtype(torch.float64)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # can modify this
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel()))
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel() + LinearKernel())) 
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
            # self.covar_module = ScaleKernel(PeriodicKernel()+MaternKernel())
            # self.covar_module = ScaleKernel(RBFKernel())
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # this is for running the notebook in our testing framework
    import os
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else num_iter


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer # for hyperparameter tuning
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=learning_rate)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose*1*((i+1)%50==0 or i==0):
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i + 1, training_iter, loss.item(),model.covar_module.base_kernel.lengthscale.item(),model.likelihood.noise.item()))
            # print('Iter %d/%d - Loss: %.3f    noise: %.3f' % (i + 1, training_iter, loss.item(),model.likelihood.noise.item()))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    return model,likelihood

def gpregression_pytorch05(X_train,y_train,num_iter=200,learning_rate=0.1,verbose=False):
    
    """
    Gaussian Process Regression implementation with the GPyTorch
    
    Input: gpregression_pytorch(X_train,y_train,num_iter,learning_rate)
    
    Retrun: trained_model, trained_likelihood

    """
    # nfeatures=X_train.shape[1]
    train_x=torch.from_numpy(np.array(X_train))
    train_y=torch.from_numpy(np.array(y_train))
    torch.set_default_dtype(torch.float64)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # can modify this
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel()))
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel() + LinearKernel())) 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
            # self.covar_module = ScaleKernel(PeriodicKernel()+MaternKernel())
            # self.covar_module = ScaleKernel(RBFKernel())
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # this is for running the notebook in our testing framework
    import os
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else num_iter


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer # for hyperparameter tuning
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=learning_rate)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose*1*((i+1)%50==0 or i==0):
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i + 1, training_iter, loss.item(),model.covar_module.base_kernel.lengthscale.item(),model.likelihood.noise.item()))
            # print('Iter %d/%d - Loss: %.3f    noise: %.3f' % (i + 1, training_iter, loss.item(),model.likelihood.noise.item()))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    return model,likelihood

def gpregression_pytorch25(X_train,y_train,num_iter=200,learning_rate=0.1,verbose=False):
    
    """
    Gaussian Process Regression implementation with the GPyTorch
    
    Input: gpregression_pytorch(X_train,y_train,num_iter,learning_rate)
    
    Retrun: trained_model, trained_likelihood

    """
    # nfeatures=X_train.shape[1]
    train_x=torch.from_numpy(np.array(X_train))
    train_y=torch.from_numpy(np.array(y_train))
    torch.set_default_dtype(torch.float64)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # can modify this
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel()))
            # self.covar_module = ScaleKernel(ScaleKernel(RBFKernel() + LinearKernel())) 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
            # self.covar_module = ScaleKernel(PeriodicKernel()+MaternKernel())
            # self.covar_module = ScaleKernel(RBFKernel())
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # this is for running the notebook in our testing framework
    import os
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else num_iter


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer # for hyperparameter tuning
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=learning_rate)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose*1*((i+1)%50==0 or i==0):
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i + 1, training_iter, loss.item(),model.covar_module.base_kernel.lengthscale.item(),model.likelihood.noise.item()))
            # print('Iter %d/%d - Loss: %.3f    noise: %.3f' % (i + 1, training_iter, loss.item(),model.likelihood.noise.item()))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    return model,likelihood



def gprediction_pytorch(model,likelihood,X_test):
    
    """
    Gaussian Process Predictions with gpr_pytorch
    
    Use: 
    model, likelihood = gpregression_pytorch(X_train,y_train,num_iter=200,learning_rate=0.1)  
    ypred, ysigma = gprediction__pytorch(model,likelihood,X_test)

    Return: ypred and ysigma numpy arrays
    
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.from_numpy(np.array(X_test))
        torch.set_default_dtype(torch.float64)
        observed_pred = likelihood(model(test_x))
        ypred = observed_pred.mean.numpy()
        ysigma = observed_pred.stddev.numpy()
    return ypred,ysigma

def gpregression_feat(Xtrain,Ytrain,Nfeature):    
    # cmean=1.0
    # cbound=[1e-3, 1e3]
    cmean=[1.0]*Nfeature
    cbound=[[1e-3, 1e3]]*Nfeature
    kernel = C(1.0, (1e-3,1e3)) * matk(cmean,cbound,1.5) + Wht(1.0, (1e-3, 1e3))  # Matern kernel
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=False)
    gpr.fit(Xtrain, Ytrain)
    return gpr

def gpregression(Xtrain,Ytrain,Nfeature):    
    cmean=1.0
    cbound=[1e-3, 1e3]
    # cmean=[1.0]*Nfeature
    # cbound=[[1e-3, 1e3]]*Nfeature
    kernel = C(1.0, (1e-3,1e3)) * matk(cmean,cbound,1.5) + Wht(1.0, (1e-3, 1e3))  # Matern kernel
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=False)
    gpr.fit(Xtrain, Ytrain)
    return gpr

def gprediction(gpnetwork,xtest):
    y_pred, sigma = gpnetwork.predict(xtest, return_std=True)
    return y_pred, sigma

def expectedimprovement(xdata,gpnetwork,ybest,itag,epsilon):
    ye_pred, esigma = gprediction(gpnetwork, xdata)
    expI = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            zzval=itag*(ye_pred[ii]-ybest)/float(esigma[ii])
            expI[ii]=itag*(ye_pred[ii]-ybest-epsilon)*norm.cdf(zzval)+esigma[ii]*norm.pdf(zzval)
        else:
            expI[ii]=0.0
    return expI

def expectedimprovement_pytorch(xdata,gp_model,gp_likelihood,ybest,itag,epsilon):

    ye_pred, esigma = gprediction_pytorch(model=gp_model,likelihood=gp_likelihood,X_test=xdata)
    expI = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            zzval=itag*(ye_pred[ii]-ybest)/float(esigma[ii])
            expI[ii]=itag*(ye_pred[ii]-ybest-epsilon)*norm.cdf(zzval)+esigma[ii]*norm.pdf(zzval)
        else:
            expI[ii]=0.0
    return expI

def probabilityOfImprovement(xdata,gpnetwork,ybest,epsilon):  
    "ybest: GPR-predicted best output property of the TRAINING set"

    ye_pred, esigma = gprediction(gpnetwork, xdata)
    poI = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            zzval=(ye_pred[ii]-ybest-epsilon)/float(esigma[ii])
            poI[ii]=norm.cdf(zzval)
        else:
            poI[ii]=0.0
    return poI

def probabilityOfImprovement_pytorch(xdata,gp_model,gp_likelihood,ybest,epsilon):  
    "ybest: GPR-predicted best output property of the TRAINING set"

    ye_pred, esigma = gprediction_pytorch(model=gp_model,likelihood=gp_likelihood,X_test=xdata)
    poI = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            zzval=(ye_pred[ii]-ybest-epsilon)/float(esigma[ii])
            poI[ii]=norm.cdf(zzval)
        else:
            poI[ii]=0.0
    return poI

# Acquisition functions
def upperConfidenceBound(xdata,gpnetwork,epsilon):
    """
        xdata: input feature vectors or PCs of the REMAINING set
        gpnetwork: GPR model
        epsilon: control exploration/exploitation. Higher epsilon means more exploration
    """
    ye_pred, esigma = gprediction(gpnetwork, xdata)
    ucb = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            ucb[ii]=(ye_pred[ii]+epsilon*esigma[ii])
        else:
            ucb[ii]=0.0
    return ucb

# Acquisition functions
def upperConfidenceBound_pytorch(xdata,gp_model,gp_likelihood,epsilon):
    """
        xdata: input feature vectors or PCs of the REMAINING set
        gpnetwork: GPR model
        epsilon: control exploration/exploitation. Higher epsilon means more exploration
    """
    ye_pred, esigma = gprediction_pytorch(model=gp_model,likelihood=gp_likelihood,X_test=xdata)
    ucb = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            ucb[ii]=(ye_pred[ii]+epsilon*esigma[ii])
        else:
            ucb[ii]=0.0
    return ucb
    
def paretoSearch(capP,search='min'):
    # Non-dominated sorting
    
    paretoIdx=[]
    F0 = []
    for i,p in enumerate(capP):
        Sp = []
        nps = 0
        for j,q in enumerate(capP):
            if i!=j:
                if search=='min':
                    compare = p < q
                elif search=='max':
                    compare = p > q
                if any(compare):
                    Sp.append(q)
                else: 
                    nps+=1
        if nps==0:
            paretoIdx.append(i)
            F0.append(p.tolist())
    F0 = np.array(F0)
    return F0, paretoIdx

def paretoOpt(capP, metric='crowdingDistance',opt='min'):
    if capP.shape[0]<=1000:
        F0, paretoIdx = paretoSearch(capP, search=opt)
    else:
        n_parts = int(capP.shape[0]//1000.)
        rem = capP.shape[0] % 1000.  
        FList = [] 
        paretoIdxList = []
        for i in range(n_parts):
            Fi, paretoIdxi = paretoSearch(capP[1000*i:1000*(i+1)], search=opt)
            FList.append(Fi)
            ar_paretoIdxi = np.array(paretoIdxi)+1000*i
            paretoIdxList.append(ar_paretoIdxi.tolist())  
        if rem>0:
            Fi, paretoIdxi = paretoSearch(capP[1000*n_parts-1:-1], search=opt)
            FList.append(Fi)
            ar_paretoIdxi = np.array(paretoIdxi)+1000*n_parts
            paretoIdxList.append(ar_paretoIdxi.tolist())  
            
        F1 = np.concatenate(FList)
        
        paretoIdx1=np.concatenate(paretoIdxList)
        F0, paretoIdxTemp = paretoSearch(F1, search=opt)
        
        paretoIdx=[]
        for a in paretoIdxTemp:
            matchingArr = np.where(capP==F1[a])[0]
            counts = np.bincount(matchingArr)
            pt = np.argmax(counts)
            paretoIdx.append(pt)

    m=F0.shape[-1]
    l = len(F0)

    ods = np.zeros(np.max(paretoIdx)+1)
    if metric == 'crowdingDistance':
        infi = 1E6
        for i in range(m):
            order = []
            sortedF0 = sorted(F0, key=lambda x: x[i])
            for a in sortedF0: 
                matchingArr = np.where(capP==a)[0]
                counts = np.bincount(matchingArr)
                o = np.argmax(counts)
                order.append(o)
            ods[order[0]]=infi
            ods[order[-1]]=infi
            fmin = sortedF0[0][i]
            fmax = sortedF0[-1][i]
            for j in range(1,l-1):
                ods[order[j]]+=(capP[order[j+1]][i]-capP[order[j-1]][i])/(fmax-fmin)
        # Impose criteria on selecting pareto points
        if min(ods[np.nonzero(ods)])>=infi:
            bestIdx = np.argmax(ods)
        else:
            if l>2: # if there are more than 2 pareto points, pick inner points with largest crowding distance (i.e most isolated)
                tempOds=copy(ods)
                for i,a in enumerate(tempOds):
                    if a>=infi: tempOds[i]=0.
                bestIdx = np.argmax(tempOds)
            else: #pick pareto point with lower index
                bestIdx = np.argmax(ods)
    elif metric == 'euclideanDistance':  # To the hypothetical point of the current data
        for i in range(m):
            order = []
            sortedF0 = sorted(F0, key=lambda x: x[i])
            for a in sortedF0:
                matchingArr = np.where(capP==a)[0]
                counts = np.bincount(matchingArr)
                o = np.argmax(counts)
                order.append(o)          
            fmin = sortedF0[0][i]
            fmax = sortedF0[-1][i]
            for j in range(0,l):
                ods[order[j]]+=((capP[order[j]][i]-fmax)/(fmax-fmin))**2
        ods = np.sqrt(ods)
        for i,a in enumerate(ods):
            if a!=0: print(i,a)
        bestIdx = np.where(ods==np.min(ods[np.nonzero(ods)]))[0][0]
    return paretoIdx,bestIdx


def next_smiles_gpr(Xtrain,Xtrain_smi,Xtrain_comp, Ytrain,Xremain,Xremain_smi,Xremain_comp,epsilon=0.01,BOmetric='crowdingDistance'):
    """
    This function will do one step of the MBO and return a SMILES for next DFT calculation and updated Xtrain and Xremain. 
    
    Use: 
    next_smi_from_gpr(Xtrain,Xtrain_smi,Xtrain_comp, Ytrain,Xremain,Xremain_smi,Xremain_comp,epsilon=0.01,BOmetric='crowdingDistance')
    
    Xtrain, Ytrain, Xremain are np array
    Xtrain_smi, Xremain_smi are python list  or np array
    Xtrain_comp, Xremain_comp are python list or np array
    
    Output: next_mol_smi,Xtrain_new,Xtrain_new_smi,Xremain_new,Xremain_new_smi
    
    """

    nobj=Ytrain.shape[1] # nobj: Number of objectives
    natom_layer = Xtrain.shape[1] # natom_layer: Number of PCs used in PCA
    gpnetworkList = []
    yt_predList = []

    for i in range(nobj):

        gpnetwork = gpregression(Xtrain, Ytrain[:,i], natom_layer)
        yt_pred, _ = gprediction(gpnetwork, Xtrain)
        yt_predList.append(yt_pred)
        gpnetworkList.append(gpnetwork)
        
    yt_pred=np.vstack((yt_predList)).T
    _, ybestloc = paretoOpt(yt_pred,metric=BOmetric,opt='max') 
    ybest = yt_pred[ybestloc]

    expIList = []
    for i in range(nobj):
        expI = expectedimprovement(Xremain, gpnetworkList[i], ybest[i], itag=1, epsilon=epsilon)
        expIList.append(expI)
    
    expI = np.vstack((expIList)).T
    # print("exit",expI)
    _, expimaxloc = paretoOpt(expI,metric=BOmetric,opt='max')
    # expImax = expI[expimaxloc]
    # print(f"Maximum expected improvemnt value = {expImax}")
    next_mol_smi=Xremain_smi[expimaxloc]    

    print("Next molecule: ",next_mol_smi)  
    Xtrain_new = np.append(Xtrain, Xremain[expimaxloc]).reshape(-1, natom_layer)
    Xtrain_new_smi = np.append(Xtrain_smi, Xremain_smi[expimaxloc])
    Xtrain_new_comp = np.append(Xtrain_comp, Xremain_comp[expimaxloc])
    print("Next molecule complexity: ",Xremain_comp[expimaxloc])
    Xremain_new=np.delete(Xremain, expimaxloc, 0)
    Xremain_new_smi=np.delete(Xremain_smi, expimaxloc)
    Xremain_new_comp=np.delete(Xremain_comp, expimaxloc)   

    return next_mol_smi,Xtrain_new,Xtrain_new_smi,Xtrain_new_comp,Xremain_new,Xremain_new_smi,Xremain_new_comp

def next_smiles_gpr_pytorch(Xtrain,Xtrain_smi,Xtrain_comp, Ytrain,Xremain,Xremain_smi,Xremain_comp,epsilon=0.01,num_iter=400,learning_rate=0.1,itag=1,BOmetric='crowdingDistance',verbose=False):
    """
    This function will do one step of the MBO and return a SMILES for next DFT calculation and updated Xtrain and Xremain. 
    
    Use: 
    next_smi_from_gpr(Xtrain,Xtrain_smi,Xtrain_comp, Ytrain,Xremain,Xremain_smi,Xremain_comp,epsilon=0.01,BOmetric='crowdingDistance')
    
    Xtrain, Ytrain, Xremain are np array
    Xtrain_smi, Xremain_smi are python list  or np array
    Xtrain_comp, Xremain_comp are python list or np array
    
    Output: next_mol_smi,Xtrain_new,Xtrain_new_smi,Xremain_new,Xremain_new_smi
    
    """

    nobj=Ytrain.shape[1] # nobj: Number of objectives
    natom_layer = Xtrain.shape[1] # natom_layer: Number of PCs used in PCA
    gpnetworkList = []
    gpnetwork_like_List = []
    yt_predList = []

    for i in range(nobj):

        model,likelihood = gpregression_pytorch(X_train=Xtrain, y_train=Ytrain[:,i],num_iter=num_iter,learning_rate=learning_rate,verbose=verbose)
        yt_pred, _ = gprediction_pytorch(model=model,likelihood=likelihood, X_test=Xtrain)
        yt_predList.append(yt_pred)
        gpnetworkList.append(model)
        gpnetwork_like_List.append(likelihood)
        
    yt_pred=np.vstack((yt_predList)).T
    _, ybestloc = paretoOpt(yt_pred,metric=BOmetric,opt='max') 
    ybest = yt_pred[ybestloc]

    expIList = []
    for i in range(nobj):
        expI = expectedimprovement_pytorch(xdata=Xremain, gp_model=gpnetworkList[i],gp_likelihood=gpnetwork_like_List[i], ybest=ybest[i], itag=itag, epsilon=epsilon)
        expIList.append(expI)
    
    expI = np.vstack((expIList)).T
    # expI[:,1] = 10*expI[:,1]
    # print("exit1",expI[:,0],"exit2",expI[:,1])
    _, expimaxloc = paretoOpt(expI,metric=BOmetric,opt='max')
    # expImax = expI[expimaxloc]
    # print(f"Maximum expected improvemnt value = {expImax}")
    next_mol_smi=Xremain_smi[expimaxloc]    

    print("Next molecule: ",next_mol_smi)  
    Xtrain_new = np.append(Xtrain, Xremain[expimaxloc]).reshape(-1, natom_layer)
    Xtrain_new_smi = np.append(Xtrain_smi, Xremain_smi[expimaxloc])
    Xtrain_new_comp = np.append(Xtrain_comp, Xremain_comp[expimaxloc])
    print("Next molecule complexity: ",Xremain_comp[expimaxloc])
    Xremain_new=np.delete(Xremain, expimaxloc, 0)
    Xremain_new_smi=np.delete(Xremain_smi, expimaxloc)
    Xremain_new_comp=np.delete(Xremain_comp, expimaxloc)   

    return next_mol_smi,Xtrain_new,Xtrain_new_smi,Xtrain_new_comp,Xremain_new,Xremain_new_smi,Xremain_new_comp

def plot_gpr_results(y_true,y_pred,y_pred_err=None,err_bar=False,label="test",color="blue",showfig=True,savefig=False,filename="gpr_pred.png",data=True):
    """
    Return r2,rmse and mae score if data = True
    Show image if showfig=True
    Save image if savefig=True
    """
    from matplotlib import pyplot as plt
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error as MAE
    mae_test = round(MAE(y_true,y_pred),2)
    r2_test = round(r2_score(y_true,y_pred),2)
    rmse_test = round(np.sqrt(MSE(y_true,y_pred)),2)

    print(f"(%s: R2 = %0.2f, RMSE = %0.2f, MAE = %0.2f)" %(label,r2_test,rmse_test,mae_test))    
    
    if showfig:
        # plt.scatter(y_true,y_pred,color=color,label=label)
        if err_bar*(y_pred_err is not None):
            plt.errorbar(y_true,y_pred,yerr=y_pred_err,color=color,fmt='o',label=label)
        plt.plot([1,2],[1,2],color='black')
        string="MAE ="+str(mae_test)
        plt.text(1.0,1.90,string)
        string="$R^2$ ="+str(r2_test)
        plt.text(1.0,1.80,string)
        string="RMSE ="+str(rmse_test)
        plt.text(1.0,1.70,string)
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.legend(loc='lower right') #best
        print("min error =", round(y_pred_err.min(),2),"max_err", round(y_pred_err.max(),2))
        if savefig:
            plt.savefig(filename,dpi=300,bbox_inches='tight')
        plt.show()



    if data:
        return r2_test,rmse_test,mae_test

def remove_corr_features(Xdata,corr_cutoff = 0.75):
    """
    This function will drop highly correlated features
    Output: a pd.Dataframe 
    """
    cor_matrix=Xdata.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_cutoff)]
    print(f"Dropped %d features with correlation coeff. > %0.2f" %(len(to_drop),corr_cutoff))

    Xdata=Xdata.drop(columns=to_drop,axis=1)
    print(f"Remaining features %d" %(Xdata.shape[1]))
    return Xdata

import os
def conformer_search(m):
    AllChem.EmbedMolecule(m)
    arr = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=20000)
    idx = np.argmin(arr, axis=0)[1]
    conf = m.GetConformers()[idx]
    m.RemoveAllConformers()
    m.AddConformer(conf)
    return m

def optimize_ETKDG(m):
    AllChem.EmbedMolecule(m)
    AllChem.EmbedMolecule(m,AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(m)
    return m

def write_mol(m):
    cwd=os.getcwd()
    name=m.GetProp('_Name')
    print(Chem.MolToMolBlock(m))
    f=open(cwd+'/data/mol/'+name+'.mol','w+')
    d=Chem.MolToMolBlock(m)
    f.write(d)
    smiles=Chem.MolToSmiles(m)
    print('\n\n>SMILES:\n',smiles,file=f)
    print('\n\n>$$$$',file=f)   
    f.close()

def output_once(m):        
    m = Chem.AddHs(m)  
    m = optimize_ETKDG(m)
    m = conformer_search(m)
    write_mol(m)
    
def ring_stats(smiles):
    
    # cwd=os.getcwd()          
    # print('\n>>> name=',name)
    print('>>> smiles=',smiles)
    m = Chem.MolFromSmiles(smiles)
    
    d=13
    nring=[0]*d
    nHring=[0]*d
    nC=[[0]*d for _ in range(d)]
    nO=[[0]*d for _ in range(d)]
    nN=[[0]*d for _ in range(d)]
    nS=[[0]*d for _ in range(d)]
    
    print("rings:")
    
    ri=m.GetRingInfo()
    for r in ri.AtomRings():
        sr=""
        n=0
        for i in r:
            sr+=m.GetAtomWithIdx(i).GetSymbol()
            n+=1
        # print("-"*10)
        print(n,sr)
        # print("-"*10)
        if(n<d):
            # print(n,d)
            nring[n]+=1
            c=sr.count('C')
            if(c<n): # hetrcycle
                nHring[n]+=1
            nC[n][c]+=1
            nO[n][sr.count('O')]+=1
            nN[n][sr.count('N')]+=1
            nS[n][sr.count('S')]+=1
            
    print("\r\nring stats:")        
      
    for n in range(3,d):
        if(nHring[n]):
            print("%d-atom heterocycles: %d" % (n,nHring[n])) 
        if(nring[n]):
            print("%d-atom rings: %d" % (n,nring[n]))
            sr=""
            for i in range(1,d):
                if(nC[n][i]):
                    sr+="C"+str(i)+":"+str(nC[n][i])+" "
                if(nO[n][i]):
                    sr+="O"+str(i)+":"+str(nO[n][i])+" "
                if(nN[n][i]):
                    sr+="N"+str(i)+":"+str(nN[n][i])+" "
                if(nS[n][i]):
                    sr+="S"+str(i)+":"+str(nS[n][i])+" "
            print("="*10)
            print(sr)
            print("="*10)
        

def main():
    print("This script has utility functions for scaling, pca, molecular descriptor generation, GPR")
    return None

if __name__== "__main__":
    main()
