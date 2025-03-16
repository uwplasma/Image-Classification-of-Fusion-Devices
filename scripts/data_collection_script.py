import sys
import os
import pandas as pd
# Get the root directory (parent of scripts)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the pyQSC directory to the system path
pyqsc_path = os.path.join(root_dir, 'pyQSC')
sys.path.append(pyqsc_path)

# Now you can import as if you are directly in the pyQSC directory
from qsc.qsc import Qsc
from qsc.plot import plot_boundary


df = pd.read_csv("data/XGStels/XGStels.csv")

#For training use [2:]
df = df.iloc[4835:].reset_index(drop=True)

# Iterate over the rows in the DataFrame
for i, row in df.iterrows():
    rc1 = row["rc1"]
    rc2 = row["rc2"]
    rc3 = row["rc3"]
    zs1 = row["zs1"]
    zs2 = row["zs2"]
    zs3 = row["zs3"]
    nfp = row["nfp"]
    etabar = row["etabar"]
    B2c = row["B2c"]
    p2 = row["p2"]
    

    try: 
        # Create Qsc instance and plot boundary
        #If want to generate stel_images for training: i+2
        stel = Qsc(rc=[1, rc1, rc2, rc3], zs=[0, zs1, zs2, zs3], nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order='r2')
        stel.plot_boundary(r=0.1, savefig=f"image{i+4835}")
        
    except Exception as e:
        print(f"Plotting failed for row {i+4835}: {e}")
        print(f"Values: {rc1}, {rc2}, {rc3}, {zs1}, {zs2}, {zs3}, {nfp}, {etabar}, {B2c}, {p2}\n")
