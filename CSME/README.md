# CSME Script

1. Run the script 'CSME_code.py' in the terminal using the commond: 


  
  - create conda environment:  conda create --name myenv --file requirements.txt
  
  - activate conda environment: conda activate myenv
  
  - run: python CSME_code.py
  
After you hit the enter, provide the address of the input file in the terminal 
    - if you want to hard code the file address modify variable inp in the CSME_code.py, for example, 
    inp = r'address of the input_csme.csv'

2. The molecule_generator.py has a GenerateSmiles() class that will generate a library of molecules (canonical SMILES) for a given core molecule and chemical building blocks.

    It takes following arguments:             

    (1) content: address of the input file with information about the core molecule and chemical building blocks
        (Note: always end file with commas so that our code know where to stop)

    (2) complexity_max: maximum CS value allowed in the library

    (3) x0 (int): CS value below which all molecules with CS < x0 are allowed in the library

    (4) beta (int or float): β is the exponential factor that is analogous to the Boltzmann factor 1/kT in thermodynamics. 
    The higher is the “temperature” (the smaller is β), the weaker is the penalization of molecular complexity. 

    (5) nmax: number of molecules in the library

    The program will save new molecules in a csv file with name: {input_csme}_mol_library.csv"

3. A sample input file (input_csme.csv) is provided in this repo
   
   The first line 'core' is followed by the SMILES string of the core molecules with growth points  indicated by R1, R2 and so on. (User can change the SMILES string and the position of R1, R2). 

   The ',,,,' separtes different sections in the input file 


   The details of the casts are followed by the line which has keyword 'group'
   
   The details of the building blocks and their weights are followed by the line which has keyword 'types'
   
   The details of the position of growth points are followed by the line which has keyword 'position'
   
   Note: always end file and each section with the commas
