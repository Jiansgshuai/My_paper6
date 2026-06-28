from pathlib import Path
from pypdf import PdfReader
p = Path('考虑车辆形态位姿一致性的路侧激光雷达车辆轨迹采集#2022211401.pdf')
reader = PdfReader(str(p))
with open('pdf_text_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(f'pages {len(reader.pages)}\n\n')
    for i in range(min(15, len(reader.pages))):
        text = reader.pages[i].extract_text() or ''
        f.write(f'--- page {i+1} ---\n')
        f.write(text[:2500])
        f.write('\n\n')
