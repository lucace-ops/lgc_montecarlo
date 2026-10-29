"""PDF report generation for L & G Consulting Portfolio Lab."""
from io import BytesIO
from datetime import datetime
import os
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from svglib.svglib import svg2rlg

NAVY=colors.HexColor("#101820"); GOLD=colors.HexColor("#D5B36A"); LIGHT=colors.HexColor("#F2F4F6")

def _table(rows,widths=None,font=7.5):
    t=Table(rows,repeatRows=1,colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.lightgrey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("PADDING",(0,0),(-1,-1),5),("FONTSIZE",(0,0),(-1,-1),font)])); return t

def build_pdf(results:dict,settings:dict,config:dict,figures:dict)->bytes:
    buffer=BytesIO(); doc=SimpleDocTemplate(buffer,pagesize=landscape(A4),rightMargin=1.2*cm,leftMargin=1.2*cm,topMargin=1.2*cm,bottomMargin=1.2*cm); styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="LGTitle",parent=styles["Title"],textColor=NAVY,fontName="Helvetica-Bold",fontSize=20,leading=24)); styles.add(ParagraphStyle(name="LGSub",parent=styles["Normal"],textColor=GOLD,fontName="Helvetica-Bold",fontSize=9,leading=12)); styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=7.5,leading=10)); story=[]
    logo_path=os.path.join(os.path.dirname(__file__),"logo.svg")
    try:
        drawing=svg2rlg(logo_path)
        if drawing:
            scale=min(1.,2.*cm/max(drawing.width,1)); drawing.scale(scale,scale); drawing.width*=scale; drawing.height*=scale; story.append(Table([[drawing,Paragraph("L &amp; G CONSULTING",styles["LGTitle"])],["",Paragraph("PORTFOLIO LAB · MONTE CARLO &amp; DECISION REPORT",styles["LGSub"])]],colWidths=[2.5*cm,20*cm]))
        else: story.append(Paragraph("◇ L &amp; G CONSULTING",styles["LGTitle"]))
    except Exception: story.append(Paragraph("◇ L &amp; G CONSULTING",styles["LGTitle"]))
    story.append(Spacer(1,.25*cm)); story.append(Paragraph(f"Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')} · Modalità {settings.get('mode','n/d')}",styles["Normal"])); story.append(Paragraph("Documento quantitativo sperimentale: non costituisce consulenza finanziaria o fiscale.",styles["Small"])); story.append(Spacer(1,.35*cm))
    story.append(Paragraph("Parametri della simulazione",styles["Heading2"])); params=[["Parametro","Valore"],["Capitale iniziale",f"€ {settings.get('initial_value',0):,.0f}"],["PAC",f"€ {settings.get('contribution',0):,.0f} · {settings.get('contribution_frequency','Mensile')}"],["Crescita annua PAC",f"{settings.get('contribution_growth_rate',0):.1%}"],["Orizzonte",f"{settings.get('years',0)} anni"],["Simulazioni",f"{settings.get('n_sims',0):,}"],["Inflazione",f"{settings.get('inflation_rate',0):.2%}"],["Motore",settings.get('method','n/d')],["Ribilanciamento",settings.get('rebalance_mode','n/d')],["Fiscalità italiana semplificata","Attiva" if settings.get('taxation_enabled') else "Disattiva"],["Commissione fissa",f"€ {settings.get('transaction_fee_fixed',0):,.2f}"],["Commissione variabile",f"{settings.get('transaction_fee_pct',0):.2f}%"],["Costo piattaforma",f"{settings.get('annual_platform_fee_pct',0):.2f}% annuo"]]; story.append(_table(params,[7*cm,9*cm],8)); story.append(Spacer(1,.4*cm))
    story.append(Paragraph("Composizione dei portafogli",styles["Heading2"])); comp=[["Portafoglio","Ticker","Peso","Classe","TER annuo","Aliquota"]]
    for p in config.get("portfolios",[]):
        for a in p.get("assets",[]): comp.append([p.get("name",""),a.get("ticker",""),f"{float(a.get('weight',0) or 0):.1f}%",a.get("asset_class",""),f"{float(a.get('ter',0) or 0):.2f}%",f"{float(a.get('tax_rate',.26) or .26):.1%}"])
    story.append(_table(comp,[4*cm,3*cm,2*cm,5*cm,2.5*cm,2.5*cm],7.2)); story.append(PageBreak())
    story.append(Paragraph("Risultati sintetici",styles["Heading2"])); rows=[["Portafoglio","Versato","Lordo mediano","Netto mediano","Netto reale","Tasse medie","Costi medi","Prob. perdita","Max DD"]]
    for name,r in results.items(): rows.append([name,f"€{r.get('total_invested',0):,.0f}",f"€{r.get('median_terminal_gross',0):,.0f}",f"€{r.get('median_terminal',0):,.0f}",f"€{r.get('median_real_terminal',0):,.0f}",f"€{r.get('taxes_paid_mean',0):,.0f}",f"€{r.get('fees_paid_mean',0):,.0f}",f"{r.get('probability_loss',0):.1%}",f"{r.get('max_drawdown_median',0):.1%}"])
    story.append(_table(rows,[3.5*cm,2.2*cm,2.7*cm,2.7*cm,2.7*cm,2.2*cm,2.2*cm,2.2*cm,2.*cm],7.)); story.append(Spacer(1,.35*cm))
    risk=[["Portafoglio","P5 netto","P95 netto","Perdita >10%","Perdita >20%","Peggior 5% medio","Underwater mediano","Sharpe","Sortino"]]
    for name,r in results.items(): risk.append([name,f"€{r.get('p5_terminal',0):,.0f}",f"€{r.get('p95_terminal',0):,.0f}",f"{r.get('probability_loss_10',0):.1%}",f"{r.get('probability_loss_20',0):.1%}",f"€{r.get('tail5_terminal_mean',0):,.0f}",f"{r.get('underwater_days_median',0)/30.4:.1f} mesi",f"{r.get('sharpe',0):.2f}",f"{r.get('sortino',0):.2f}"])
    story.append(_table(risk,[3.4*cm,2.3*cm,2.3*cm,2.1*cm,2.1*cm,2.6*cm,2.7*cm,1.7*cm,1.7*cm],6.8))
    for title,fig in figures.items():
        if fig is None: continue
        story.append(PageBreak()); story.append(Paragraph(title,styles["Heading2"]))
        try: img=pio.to_image(fig,format="png",width=1200,height=600,scale=1); story.append(Image(BytesIO(img),width=24*cm,height=12*cm))
        except Exception as exc: story.append(Paragraph(f"Grafico non incorporato ({title}): {exc}",styles["Small"]))
    story.append(PageBreak()); story.append(Paragraph("Stress, proxy e avvertenze",styles["Heading2"])); any_warn=False
    for name,r in results.items():
        for w in r.get("warnings",[]): any_warn=True; story.append(Paragraph(f"• <b>{name}</b>: {w}",styles["Small"])); story.append(Spacer(1,.08*cm))
    for w in settings.get("market_warnings",[]): any_warn=True; story.append(Paragraph("• "+str(w),styles["Small"])); story.append(Spacer(1,.08*cm))
    if not any_warn: story.append(Paragraph("Nessuna avvertenza specifica per questa simulazione.",styles["Small"]))
    story.append(Spacer(1,.35*cm)); story.append(Paragraph("Analisi decisionali disponibili nell'app",styles["Heading2"])); story.append(Paragraph("Obiettivo finanziario e PAC necessario, benchmark, sequence-of-returns risk, sensibilità, confronto prima/dopo e decumulo sono moduli interattivi eseguiti su richiesta per evitare di rallentare la simulazione principale.",styles["Small"])); story.append(Spacer(1,.35*cm)); story.append(Paragraph("Metodologia e limiti",styles["Heading2"])); notes=["I rendimenti storici sono convertiti in EUR e utilizzati per bootstrap a blocchi o per calibrare il modello parametrico.","La fiscalità è una stima semplificata: non gestisce ogni peculiarità fiscale, compensazione delle minusvalenze o regime dell'intermediario.","TER, commissioni e costo piattaforma riducono i percorsi secondo i parametri inseriti.","Gli stress storici sono scenari espliciti; quando manca lo storico possono essere usati proxy dichiarati.","Il decumulo usa i rendimenti simulati del portafoglio ed è un'analisi di scenario, non un piano pensionistico certificato.","Le simulazioni non sono previsioni né garanzie di rendimento futuro."]
    for n in notes: story.append(Paragraph("• "+n,styles["Small"])); story.append(Spacer(1,.1*cm))
    doc.build(story); return buffer.getvalue()