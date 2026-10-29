"""Branded PDF report generation for L & G Consulting Portfolio Lab."""
from __future__ import annotations
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

NAVY=colors.HexColor("#101820")
GOLD=colors.HexColor("#D5B36A")
LIGHT=colors.HexColor("#F4F6F8")

def _fmt_money(v): return f"€{float(v):,.0f}".replace(",", ".")
def _fmt_pct(v): return f"{float(v):.1%}"

def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GOLD); canvas.setLineWidth(0.8)
    canvas.line(1.4*cm,1.05*cm,landscape(A4)[0]-1.4*cm,1.05*cm)
    canvas.setFillColor(colors.HexColor("#59636e")); canvas.setFont("Helvetica",7.5)
    canvas.drawString(1.4*cm,0.68*cm,"L & G Consulting · Portfolio Lab")
    canvas.drawRightString(landscape(A4)[0]-1.4*cm,0.68*cm,f"Pagina {doc.page}")
    canvas.restoreState()

def build_pdf(results: dict, settings: dict, config: dict, figures: dict | None = None) -> bytes:
    figures=figures or {}
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=landscape(A4),rightMargin=1.35*cm,leftMargin=1.35*cm,topMargin=1.2*cm,bottomMargin=1.35*cm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LGTitle",parent=styles["Title"],textColor=NAVY,fontName="Helvetica-Bold",fontSize=21,leading=24,spaceAfter=3))
    styles.add(ParagraphStyle(name="LGKicker",parent=styles["Normal"],textColor=GOLD,fontName="Helvetica-Bold",fontSize=9,leading=11,spaceAfter=2))
    styles.add(ParagraphStyle(name="LGH2",parent=styles["Heading2"],textColor=NAVY,fontSize=13,leading=16,spaceBefore=5,spaceAfter=6))
    styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8,leading=10))
    styles.add(ParagraphStyle(name="Disclaimer",parent=styles["BodyText"],fontSize=7.5,leading=10,textColor=colors.HexColor("#4B5563")))
    story=[]
    logo_path=os.path.join(os.path.dirname(__file__),"logo.svg")
    logo=None
    try: logo=svg2rlg(logo_path)
    except Exception: logo=None
    if logo:
        scale=min(1.6*cm/max(logo.height,1),2.0*cm/max(logo.width,1)); logo.scale(scale,scale); logo.width*=scale; logo.height*=scale
        head=Table([[logo,Paragraph("L &amp; G CONSULTING",styles["LGTitle"])],["",Paragraph("PORTFOLIO MONTE CARLO LAB · ANALISI QUANTITATIVA",styles["LGKicker"])]],colWidths=[2.3*cm,20*cm])
        head.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),1)])); story.append(head)
    else:
        story.append(Paragraph("L &amp; G CONSULTING",styles["LGTitle"])); story.append(Paragraph("PORTFOLIO MONTE CARLO LAB",styles["LGKicker"]))
    story.append(Spacer(1,.15*cm))
    story.append(Paragraph(f"Report generato il {datetime.now().strftime('%d/%m/%Y %H:%M')} · valori espressi in EUR salvo diversa indicazione.",styles["Small"]))
    story.append(Spacer(1,.35*cm))

    story.append(Paragraph("1. Parametri della simulazione",styles["LGH2"]))
    rows=[["Parametro","Valore","Parametro","Valore"],
          ["Capitale iniziale",_fmt_money(settings.get("initial_value",0)),"PAC",f"{_fmt_money(settings.get('contribution',0))} · {settings.get('contribution_frequency','-')}"],
          ["Orizzonte",f"{settings.get('years','-')} anni","Simulazioni",f"{int(settings.get('n_sims',0)):,}".replace(",",".")],
          ["Metodo",settings.get("method","-"),"Distribuzione",settings.get("distribution","-")],
          ["Storico usato",f"{settings.get('history_years','-')} anni","Inflazione",_fmt_pct(settings.get("inflation_rate",0))],
          ["Risk-free",_fmt_pct(settings.get("risk_free_rate",0)),"Soglia rovina",_fmt_pct(settings.get("ruin_threshold",0))]]
    if settings.get("method")=="Bootstrap storico": rows.append(["Block bootstrap",f"{settings.get('block_length',1)} mesi","Conversione valuta","Rendimenti normalizzati in EUR"])
    else: rows.append(["Gradi libertà Student-t",str(settings.get("degrees_of_freedom","-")),"Conversione valuta","Rendimenti normalizzati in EUR"])
    t=Table(rows,colWidths=[4.6*cm,7.2*cm,4.6*cm,7.2*cm],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("BACKGROUND",(0,1),(-1,-1),LIGHT),("GRID",(0,0),(-1,-1),.35,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)])); story.append(t)

    story.append(Paragraph("2. Composizione dei portafogli",styles["LGH2"]))
    prow=[["Portafoglio","Ticker","Classe","Peso"]]
    for p in config.get("portfolios",[]):
        for a in p.get("assets",[]):
            prow.append([p.get("name","Portafoglio"),str(a.get("ticker","")),str(a.get("asset_class","")),f"{float(a.get('weight',0)):.1f}%"])
    pt=Table(prow,repeatRows=1,colWidths=[7*cm,4*cm,9*cm,3*cm])
    pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),4)])); story.append(pt)

    story.append(PageBreak()); story.append(Paragraph("3. Risultati comparativi",styles["LGH2"]))
    mrows=[["Portafoglio","Versato","Mediana finale","P5","P95","Reale mediano","Prob. perdita","Prob. rovina","Max DD"]]
    for name,r in results.items():
        mrows.append([name,_fmt_money(r.get("total_invested",0)),_fmt_money(r.get("median_terminal",0)),_fmt_money(r.get("p5_terminal",0)),_fmt_money(r.get("p95_terminal",0)),_fmt_money(r.get("median_real_terminal",0)),_fmt_pct(r.get("probability_loss",0)),_fmt_pct(r.get("probability_ruin",0)),_fmt_pct(r.get("max_drawdown_median",0))])
    mt=Table(mrows,repeatRows=1,colWidths=[5.3*cm,2.7*cm,3*cm,2.7*cm,2.7*cm,3*cm,2.5*cm,2.5*cm,2.3*cm])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),7.2),("PADDING",(0,0),(-1,-1),4)])); story.append(mt); story.append(Spacer(1,.35*cm))
    rrows=[["Portafoglio","VaR 95% m","CVaR 95% m","VaR 99% m","CVaR 99% m","Sharpe","Sortino","Vol ann."]]
    for name,r in results.items():
        rrows.append([name,_fmt_pct(r.get("var95_monthly",0)),_fmt_pct(r.get("cvar95_monthly",0)),_fmt_pct(r.get("var99_monthly",0)),_fmt_pct(r.get("cvar99_monthly",0)),f"{r.get('sharpe',0):.2f}",f"{r.get('sortino',0):.2f}",_fmt_pct(r.get("annualized_volatility",0))])
    rt=Table(rrows,repeatRows=1,colWidths=[6*cm,3*cm,3*cm,3*cm,3*cm,2.3*cm,2.3*cm,3*cm])
    rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),7.5),("PADDING",(0,0),(-1,-1),4)])); story.append(rt)

    for title,fig in figures.items():
        if fig is None: continue
        try:
            png=pio.to_image(fig,format="png",width=1300,height=650,scale=1)
            story.append(PageBreak()); story.append(Paragraph(title,styles["LGH2"])); story.append(Image(BytesIO(png),width=24.5*cm,height=12.25*cm))
        except Exception as exc:
            story.append(Paragraph(f"Grafico non incorporato ({title}): {exc}",styles["Small"]))

    story.append(PageBreak()); story.append(Paragraph("4. Stress storici, proxy e coperture",styles["LGH2"]))
    historical=settings.get("historical_events",[]) or []
    if historical:
        story.append(Paragraph("Stress selezionati: "+", ".join(historical)+".",styles["Small"]))
    else: story.append(Paragraph("Nessuno stress storico selezionato.",styles["Small"]))
    for name,r in results.items():
        story.append(Spacer(1,.15*cm)); story.append(Paragraph(f"<b>{name}</b>",styles["Small"]))
        hedge=r.get("hedging",{})
        if settings.get("include_put_hedge"):
            story.append(Paragraph(f"Put teorica: premio medio cumulato {_fmt_money(hedge.get('premium_paid',0))}; payoff medio {_fmt_money(hedge.get('payoff_received',0))}; strike {float(hedge.get('moneyness',0)):.0%} OTM; scadenza {hedge.get('maturity_months','-')} mesi; IV {float(hedge.get('implied_volatility',0)):.0%}; quota coperta {float(hedge.get('hedge_fraction',0)):.0%}.",styles["Small"]))
        for w in r.get("warnings",[]):
            story.append(Paragraph("• "+str(w),styles["Small"]))

    story.append(PageBreak()); story.append(Paragraph("5. Metodologia e limiti",styles["LGH2"]))
    notes=[
        "I rendimenti degli strumenti quotati in valute diverse dall'euro vengono convertiti in EUR prima della stima dei rendimenti mensili.",
        "Il bootstrap a blocchi conserva sequenze di mesi consecutivi; il modello parametrico usa media e matrice di covarianza storiche. La Student-t aumenta la probabilità di code pesanti rispetto alla normale.",
        "Gli stress storici sono scenari espliciti: la sequenza mensile osservata nella finestra storica sostituisce i rendimenti simulati. Non sono eventi casuali inseriti con frequenza arbitraria.",
        "Se un ETF non possiede dati nella finestra storica, uno stress può essere ricostruito solo con un proxy scelto dall'utente. In tal caso il report lo segnala come PROXY_RAPPRESENTATIVO e non come performance storica effettiva del ticker.",
        "La put è valutata con Black-Scholes usando parametri ipotetici; non include smile/skew, costi di transazione, liquidità, fiscalità o slippage e non equivale a una strategia realmente negoziata.",
        "VaR, CVaR, Sharpe, Sortino, drawdown e probabilità di perdita dipendono dal campione storico e dalle ipotesi. Non sono garanzie né raccomandazioni di investimento."
    ]
    for n in notes: story.append(Paragraph("• "+n,styles["Disclaimer"])); story.append(Spacer(1,.12*cm))
    doc.build(story,onFirstPage=_header_footer,onLaterPages=_header_footer)
    return buf.getvalue()