# PRODUCES ONE SET OF STELLARATOR IMAGES (3 IMAGES)
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

row_index = 4908  # Change this to the desired row index - 2 
row = df.iloc[row_index]

# Extract parameters
rc1, rc2, rc3 = row["rc1"], row["rc2"], row["rc3"]
zs1, zs2, zs3 = row["zs1"], row["zs2"], row["zs3"]
nfp, etabar, B2c, p2 = row["nfp"], row["etabar"], row["B2c"], row["p2"]

try:
    # Create Qsc instance and plot boundary
    # Fix the NFP to a specific number
    # Change rc1 and zs1 very little --> impact is alot 
    # For rc2, zs2, rc3, zs3 --> impact is very little, can change more
    stel = Qsc(rc=[1, rc1, rc2, rc3], zs=[0, zs1, zs2, zs3], nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order='r2')
    stel.plot_boundary(r=0.1, savefig=f"test_image{row_index + 2}")
except Exception as e:
    print(f"Plotting failed for row {row_index + 2}: {e}")
    print(f"Values: {rc1}, {rc2}, {rc3}, {zs1}, {zs2}, {zs3}, {nfp}, {etabar}, {B2c}, {p2}\n")