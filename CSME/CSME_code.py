import time
from tqdm import tqdm 
from random import choice, random
import numpy as np
import os, sys
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem.Fragments import fr_nitro,fr_COO,fr_Ar_OH,fr_Al_OH

start_time=time.time()

from syba.syba import SybaClassifier
syba = SybaClassifier()
syba.fitDefaultScore()

class GenerateSmiles: 
    def __init__(self,content,complexity_max=8,x0=3,beta=4,nmax=2000,exp_table=None):
        try:
            f=open(content,'r')
            self.lines=f.readlines()  
            self.name = os.path.splitext(os.path.basename(f.name))[0] 
            self.exetension = os.path.splitext(os.path.basename(f.name))[1]
            self.max_complexity = complexity_max
            self.beta = beta
            self.max_molecules = nmax
            self.x0 = x0
            self.is_exp_table=0
            if exp_table is not None:
                self.etable=exp_table
                self.is_exp_table=1
            else:
                print("Precalculate exponentials for faster calculations")
        
        except FileNotFoundError:
            print("File not found.\n Give a csv file path with an extension")
            sys.exit(0)
            
        
    def get_file_name(self):
        """
        Return the file name and extension
        """
        try:
            return self.name,self.exetension
        except AttributeError:
            print("File not found")
            return None,None

    def line2list(self):
        lines_as_list = []
        for row in self.lines:
            lines_as_list.append(row.rstrip().split(','))
        return lines_as_list

    def get_keys_idx(self,key):
        for i,row in enumerate(self.line2list()):
            if key in row:
                return i
    
    def get_core(self):
        core_idx=self.get_keys_idx(key="core")
        return self.line2list()[core_idx][1]

    def get_casts(self):
        group_idx=self.get_keys_idx(key="group")
        core_group=[]
        core_cast=[]
        unique_groups=[]
        for i in range(group_idx+1,len(self.line2list())):
            group=self.line2list()[i][0]
            if len(group):
                core_group.append(group)
                core_cast.append(self.line2list()[i][1])
                if group not in unique_groups:
                    unique_groups.append(group)
            else:
                return core_group,core_cast,unique_groups

    def get_types_weights(self):
        types_idx=self.get_keys_idx(key="types")
        group_types=[]
        weights_frac=[]
        for i in range(types_idx+1,len(self.line2list())):
            group=self.line2list()[i][0]
            if len(group):
                group_types.append(group)
                wt=float(self.line2list()[i][1])
                weights_frac.append(wt)
            else:
                weights_frac=[weight/sum(weights_frac) for weight in weights_frac]
                return group_types, weights_frac
   
    def get_position(self):
        """
        always end file with commas so that our code know where to stop
        """
        position_idx=self.get_keys_idx(key="position")
              
        group_symm=[]
        core_groups=[]
        count=0
        for i in range(position_idx+1,len(self.line2list())):
            group=self.line2list()[i][0]
            
            if len(group):
                group_symm.append(self.line2list()[i][1])
                core_groups.append(self.line2list()[i][2])
                count+=1
            else:
                return group_symm,core_groups

    def select_core_type(self,grp):
        """
        This function will select one grp randomly for the substitution given grp (R1 or R2 core_group)
        """
        core_groups,core_types,_= self.get_casts()
        indx=[i for i,core_grp in enumerate(core_groups) if core_grp==grp]
        i = choice(indx)
        return core_types[i]
    
    def get_symm_cast(self,grp):
        """
        This function will return the possible symmerties of the R1 and R2 given in the input file
        """
        group_symm,groups = self.get_position()
        symm=[group_symm[i] for i,group in enumerate(groups) if grp==group]
        return symm

    def fill_in_the_cast_once(self,fill_,type_choice,complexity):
        """
        Fill X or Y once in the cast
        type_choice (cast) is the output of select_core_type (R1 or R2)
        return updated type_choice and complexity
        """
        type_choice = type_choice
        grp_type, wt_type = self.get_types_weights()
        complexity = complexity

        # Find the index of X or Y
        if (fill_=="Y"):
            m=type_choice.find('Y',0)
        else:
            m=type_choice.find('X',0)
        
        # if X or Y exist then fill them
        if(m>=0):  
            
            # randomly choose a substituent group
            q=np.random.choice(grp_type,None,True,wt_type)  

            if(fill_=="Y" and q=="[H]"): # Y cannot be [H], so instead choose C
                q=grp_type[1]

            # Now fill X or Y in the type_choice     
            type_choice=type_choice[0:m]+q+type_choice[m+1:]  
         
            # if q is not H, then increace complexity 
            if(q!="[H]"):
                complexity+=1

            return type_choice,complexity
        # No more X or Y left in the type_choice, return without any changes
        return type_choice, complexity

    def fill_in_the_cast_all(self,type_choice,complexity):
        """
        Fill all X and Y in the cast
        type_choice (cast) is the output of select_core_type (R1 or R2)
        return updated type_choice and complexity
        """
        type_choice = type_choice
        count_X=type_choice.count("X")
        count_Y=type_choice.count("Y")
        
        complexity = complexity

        # Keep filling X and Y until until count_X = 0 and count_Y = 0
        while (count_X or count_Y):
            # fill X 
            if count_X:
                type_choice,complexity=self.fill_in_the_cast_once(fill_="X",type_choice=type_choice,complexity=complexity)
                count_X=type_choice.count("X") # count remaining X
            # fill Y
            if count_Y:
                type_choice,complexity=self.fill_in_the_cast_once(fill_="Y",type_choice=type_choice,complexity=complexity)
                count_Y=type_choice.count("Y")
            
        return type_choice, complexity

    def select_subti_group(self,grp,core,complexity):
        """
        This function will update the given smiles with substituent groups
        """
        grp=grp
        layer=core
        complexity=complexity

        u = layer.find("("+grp+")",0) 
        if (u < 0):
            return layer, complexity
        else: # if R1 or R2 present then start filling it.
            cast = self.select_core_type(grp=grp)

            if (cast!="[H]"):
                complexity+=1

            cast,complexity = self.fill_in_the_cast_all(type_choice=cast,complexity=complexity)
            # add filled cast to the layer
            layer = layer[0:u+1]+cast+")"+layer[u+len(grp)+2:]  
        return layer,complexity
    
    def generate_smiles_once(self):
        """ 
        This function will generate a single smiles string given the scaffold/core smiles with R1,R2
        """
        complexity=0
        layer = self.get_core()
        _,_,unique_groups=self.get_casts()
        
        for grp in unique_groups:
            # get the symmetry of the grp
            symm = self.get_symm_cast(grp=grp)
            # symm will let's us know how many grp are there 
            # for example, if len(symm)=2 (R1 occurs in 2 place in the core) then do select_Subti_group 2 times
            for i, _  in enumerate(symm):
                layer,complexity=self.select_subti_group(grp=grp,core=layer,complexity=complexity)

        return layer, complexity


    def select_complexity(self,x):
        """
        if x < x0, them always accept
        if x0 < x < max_complexity then accept with probability (Like MC, random walk)
        """
        if x<=self.x0:
            return 1    
        elif x <= self.max_complexity:
            if self.is_exp_table:
                if x-self.x0==1:
                    return random() < self.etable[0]
                if x-self.x0==2:
                    return random() < self.etable[1]
                if x-self.x0==3:
                    return random() < self.etable[2]                                                
                if x-self.x0==4:
                    return random() < self.etable[3]
                if x-self.x0==5:
                    return random() < self.etable[4] 
            # else:
            #     dx = (x-self.x0)/(self.max_complexity-self.x0)
            #     # Calculations of exp are too expensive, use precalculated table
            #     if x<=self.max_complexity:
            #         if random()< np.exp(-self.beta*dx):
            #             return 1
        else:
            return 0


    
    def accept_mol(self):
        """
        This function will accept=1/reject=0 the new smiles
        Return: rdkit cleaned smiles, accepted complexity or 0 (reject)
        """
        decision=0
        n_atoms=0
        smiles,complexity=self.generate_smiles_once()
        m = Chem.MolFromSmiles(smiles)
        if m is not None:

            n_nitro,n_coo,n_ar_oh,n_al_oh=fr_nitro(m),fr_COO(m),fr_Ar_OH(m),fr_Al_OH(m)
            
            smiles=Chem.MolToSmiles(m)
            n_atoms = m.GetNumAtoms()
            syba_score = syba.predict(smiles)

            conditions=[
                        n_atoms < 30,
                        n_nitro < 2, # max 1 NO2 group
                        n_coo < 1, # no carboxylic acids 
                        n_ar_oh < 1, # no phenols 
                        n_al_oh < 1, # no aliphatic alchols
                        syba_score > 10
                        ]

            if all(conditions): # add more conditions like only 1CN and 1NO2
                decision=self.select_complexity(x=complexity)

        return decision,smiles, complexity, n_atoms
        
    def generate_library(self):
        
        nmax=self.max_molecules
        start=0
        mol_set=[]
        n_complexity=[0]*(self.max_complexity+1)
        file_name,_=self.get_file_name()
        print("Writing", file_name+"_mol_library.csv and "+file_name+"_complex_histogram.csv in ", os.getcwd())
        
        pbar=tqdm(total=nmax)
        
    
        while(start<nmax):
            keep,smiles,complexity,natoms=self.accept_mol()
        
            if keep and (smiles not in mol_set):
                
                if start==0:
                    print("Number,SMILES,Complexity,Natoms",file=open(file_name+'_mol_library.csv','w')) 

                print("%s,%s,%d,%d" %(start+1,smiles,complexity,natoms),file=open(file_name+'_mol_library.csv', 'a'))
                start+=1
                pbar.update(1)
                mol_set.append(smiles)
                n_complexity[complexity]+=1
        
        pbar.close()
        del mol_set

        for i in range(self.max_complexity+1):
            if i==0:
                print("Complexity,frequency_fraction",file=open(file_name+'_complex_histogram.csv', 'w'))
            print("%d,%3e" %(i,float(n_complexity[i])/nmax),file=open(file_name+'_complex_histogram.csv', 'a'))
        
        plt.figure()
        plt.scatter(np.arange(len(n_complexity)),np.array(n_complexity)/nmax,color="blue")
        plt.plot(np.arange(len(n_complexity)),np.array(n_complexity)/nmax,linewidth=1.25,color="blue")
        plt.xlabel("Complexity")
        plt.ylabel("Fraction")
        plt.savefig("Figure_"+file_name+"_complex_histogram.png",dpi=300)
        # plt.show()


            
def main():
    """
    content: file path with extension
    complexity_max: Maximum modifications with non-H groups
    x0: accept all smiles with complexity <= x0
    beta: exp(-beta*dx/dx0)
            Higher beta will allow fewer smiles with higher complexity
            Smaller beta will allow more smiles with higher complexity
    nmax: Number of smiles in the library
    """
    inp=input(f" Give file path with file extension: \n example 'input_csme.csv'")
    x0=3
    beta=6
    complexity_max=8
    nmax=100000
    
    if complexity_max-x0==5:
        dx=np.arange(x0+1,complexity_max+1,1)
        dx=(dx-x0)/(dx[-1]-x0)
        dx=-beta*dx
        exp_table=np.exp(dx)

        trial=GenerateSmiles(content=inp,complexity_max=complexity_max,x0=x0,beta=beta,nmax=nmax,exp_table=exp_table)
        trial.generate_library()
    else:
        print("Update exp_table in the Class")

    


if __name__ == "__main__":
    main()
    print("--- Library Generated in %0.2f minutes ---" %((time.time()-start_time)/60))
