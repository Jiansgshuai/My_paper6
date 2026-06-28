from pathlib import Path
from pypdf import PdfReader
import sys
p = Path('考虑车辆形态位姿一致性的路侧激光雷达车辆轨迹采集#2022211401.pdf')
r = PdfReader(str(p))
print(f'pages {len(r.pages)}\n')
for i in range(min(15, len(r.pages))):
    text = r.pages[i].extract_text() or ''
    print('--- page', i+1, '---')
    print(text[:2500])
    print()