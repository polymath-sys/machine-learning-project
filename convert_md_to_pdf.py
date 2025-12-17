#!/usr/bin/env python3
"""Convert markdown files to PDF using reportlab."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY


def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    print(f"Converting: {md_file}")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    doc = SimpleDocTemplate(pdf_file, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#0033cc'),
        spaceAfter=15,
        spaceBefore=10
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#0066ff'),
        spaceAfter=10,
        spaceBefore=8
    ))
    
    styles.add(ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY
    ))
    
    elements = []
    lines = content.split('\n')
    
    for line in lines:
        if line.startswith('# '):
            elements.append(Paragraph(line[2:], styles['CustomTitle']))
            elements.append(Spacer(1, 0.15*inch))
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:], styles['CustomHeading2']))
            elements.append(Spacer(1, 0.1*inch))
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:], styles['Heading3']))
            elements.append(Spacer(1, 0.08*inch))
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            bullet = line.strip()[2:]
            elements.append(Paragraph(f'• {bullet}', styles['CustomNormal']))
        elif line.strip().startswith('['):
            # Handle checkboxes
            if '[x]' in line.lower() or '✓' in line or '✅' in line:
                text = line.replace('[x]', '✓').replace('[X]', '✓')
            else:
                text = line.replace('[ ]', '☐')
            elements.append(Paragraph(f'• {text[3:] if line.startswith("[") else text}', styles['CustomNormal']))
        elif line.strip() and not line.strip().startswith('|'):
            elements.append(Paragraph(line, styles['CustomNormal']))
        elif not line.strip():
            elements.append(Spacer(1, 0.08*inch))
    
    doc.build(elements)
    print(f"  ✅ Created: {pdf_file}")


if __name__ == "__main__":
    os.chdir("/Users/adityapandey/Downloads/Drone-Guard-main")
    
    files = [
        ("VIVA_PREPARATION_REPORT.md", "VIVA_PREPARATION_REPORT.pdf"),
        ("VIVA_CHEAT_SHEET.md", "VIVA_CHEAT_SHEET.pdf"),
        ("README_VIVA.md", "README_VIVA.pdf"),
        ("QUICK_START_VIVA.txt", "QUICK_START_VIVA.pdf"),
        ("START_HERE.md", "START_HERE.pdf"),
        ("PRESENTATION_WITH_METRICS_VERIFIED.md", "PRESENTATION_WITH_METRICS_VERIFIED.pdf"),
    ]
    
    print("\n" + "=" * 60)
    print("📄 CONVERTING MARKDOWN FILES TO PDF")
    print("=" * 60 + "\n")
    
    success = 0
    for md_file, pdf_file in files:
        if os.path.exists(md_file):
            try:
                markdown_to_pdf(md_file, pdf_file)
                success += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"⚠️  Not found: {md_file}")
    
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: {success}/{len(files)} files converted to PDF")
    print("=" * 60 + "\n")
