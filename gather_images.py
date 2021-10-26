import pandas as pd
from pathlib import Path
import shutil
import sys

src_path = Path(sys.argv[1])
base_path = Path(__file__).parent
dest_path = base_path / "raw_images"

df = pd.read_csv(base_path / "mcf10a_cycif_minerva_map.csv")

def format_src_path(r):
    P = r["source_plate"]
    W = r["source_well"]
    F = r["source_field"]
    return f"Plate{P}/{W}_fld{F}.tif"

def format_dest_path(r):
    R = r["minerva_row"]
    C = r["minerva_col"]
    return f"{R}_{C}.tif"

df["src_path"] = src_path / df.apply(format_src_path, axis="columns")
df["dest_path"] = dest_path / df.apply(format_dest_path, axis="columns")

print("Copying images")
dest_path.mkdir(exist_ok=True)
for r in df.itertuples():
    print(f"...{str(r.src_path)[-30:]} -> ...{str(r.dest_path)[-20:]}")
    shutil.copyfile(r.src_path, r.dest_path)
