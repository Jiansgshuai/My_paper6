from pathlib import Path
from pypdf import PdfReader
import sys
p = Path('考虑车辆形态位姿一致性的路侧激光雷达车辆轨迹采集#2022211401.pdf')
r = PdfReader(str(p))
sys.stdout.write(f"pages {len(r.pages)}\n\n")
for i in range(min(15, len(r.pages))):
    text = r.pages[i].extract_text() or ""
    sys.stdout.write(f"--- page {i+1} ---\n")
    sys.stdout.write(text[:2500] + "\n\n")
