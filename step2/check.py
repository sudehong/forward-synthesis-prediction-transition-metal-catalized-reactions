from rdkit import Chem

# from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
import  pandas as pd

df = pd.read_csv('12.csv',)
print(df.iloc[:,0])
mols = []
for i in df.iloc[:,0].tolist():
    smi0 = i
    smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi0))  #  normalize smiles
    m1 = Chem.MolFromSmiles(smi1)
    m1H = Chem.AddHs(m1)
    mols.append(m1H)
    print('=========================')
    for a1 in m1H.GetAtoms():
        print(a1.GetSymbol(), a1.GetIdx(), a1.GetAtomicNum(), a1.GetHybridization())
        for a2 in a1.GetNeighbors():
            print('neighbors', a2.GetSymbol(), a2.GetIdx(), a2.GetAtomicNum(), a2.GetHybridization())
            b12 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
            print(b12.GetBondType(), b12.GetBondTypeAsDouble())


img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(700,700),
    legends=['' for x in mols]
    ,returnPNG=False
)
img.save('tail5.jpg')



# for a1 in m1H.GetAtoms():
#     print(a1.GetSymbol(),a1.GetIdx(),a1.GetAtomicNum(),a1.GetHybridization())