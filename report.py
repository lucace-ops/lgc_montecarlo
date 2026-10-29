"""PDF report generation for L & G Consulting."""
from io import BytesIO
from datetime import datetime
import os
import numpy as np
import pandas as pd
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Table as RLTable
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

NAVY = colors.HexColor("#101820")
GOLD = colors.HexColor("#D5B36A")

def build_pdf(results: dict, settings: dict, config: dict, figures: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=1.4*cm,
                            leftMargin=1.4*cm, topMargin=1.4*cm, bottomMargin=1.4*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LGTitle", parent=styles["Title"], textColor=NAVY,
                              fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=4))
    styles.add(ParagraphStyle(name="LGSub", parent=styles["Normal"], textColor=GOLD,
                              fontName="Helvetica-Bold", fontSize=10, leading=13))
    story = []
    logo_path = os.path.join(os.path.dirname(__file__), "logo.svg")
    # Embed the supplied SVG logo in the PDF using svglib.
    try:
        logo_drawing = svg2rlg(logo_path)
        if logo_drawing:
            scale = min(2.0, 2.0*cm / max(logo_drawing.width, 1))
            logo_drawing.width *= scale
            logo_drawing.height *= scale
            logo_drawing.scale(scale, scale)
            story.append(RLTable([[logo_drawing, Paragraph("L &amp; G CONSULTING", styles["LGTitle"])],
                                  ["", Paragraph("PORTFOLIO MONTE CARLO &amp; STRESS TEST REPORT", styles["LGSub"])]],
                                 colWidths=[2.5*cm, 17*cm], style=TableStyle([
                                     ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                                     ("LEFTPADDING", (0,0), (-1,-1), 0),
                                     ("RIGHTPADDING", (0,0), (-1,-1), 8),
                                     ("BOTTOMPADDING", (0,0), (-1,-1), 2)])))
        else:
            story.append(Paragraph("◇ L &amp; G CONSULTING", styles["LGTitle"]))
    except Exception:
        story.append(Paragraph("◇ L &amp; G CONSULTING", styles["LGTitle"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')} — documento di analisi quantitativa, non consulenza finanziaria.", styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Parametri della simulazione", styles["Heading2"]))
    setting_rows = [["Parametro", "Valore"], ["Capitale iniziale", f"€ {settings['initial_value']:,.0f}"],
                    ["Orizzonte", f"{settings['years']} anni"], ["Simulazioni", f"{settings['n_sims']:,}"],
                    ["Inflazione ipotizzata", f"{settings['inflation_rate']:.2%}"],
                    ["Cigni neri sintetici", "Attivi" if settings["include_black_swans"] else "Disattivi"],
                    ["Overlay put parametrico", "Attivo" if settings["include_put_hedge"] else "Disattivo"]]
    t = Table(setting_rows, colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                           ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),("PADDING",(0,0),(-1,-1),6)]))
    story.append(t); story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Confronto dei portafogli", styles["Heading2"]))
    rows = [["Portafoglio", "Mediana finale", "5° percentile", "95° percentile", "VaR terminale", "CVaR terminale", "Max DD mediano", "Underwater mediano"]]
    for name, r in results.items():
        rows.append([name, f"€{r['median_terminal']:,.0f}", f"€{r['p5_terminal']:,.0f}",
                     f"€{r['p95_terminal']:,.0f}", f"{r['var']:.1%}", f"{r['cvar']:.1%}",
                     f"{r['max_drawdown_median']:.1%}", f"{r['underwater_days_median']/252:.1f} anni"])
    tbl = Table(rows, repeatRows=1, colWidths=[4.1*cm,2.5*cm,2.5*cm,2.5*cm,2.1*cm,2.1*cm,2.3*cm,3.0*cm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("GRID",(0,0),(-1,-1),0.35,colors.lightgrey),("PADDING",(0,0),(-1,-1),5),
                             ("FONTSIZE",(0,0),(-1,-1),7.5)]))
    story.append(tbl)
    for title, fig in figures.items():
        if fig is None:
            continue
        try:
            img_bytes = pio.to_image(fig, format="png", width=1200, height=600, scale=1)
            image_buffer = BytesIO(img_bytes)
            story.append(PageBreak())
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(image_buffer, width=24*cm, height=12*cm))
        except Exception:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(f"Grafico '{title}' non incorporato: installare kaleido se si desidera l'esportazione grafica raster.", styles["Italic"]))
    story.append(PageBreak())
    story.append(Paragraph("Metodologia e limiti", styles["Heading2"]))
    notes = [
        "I rendimenti sono ricampionati per blocchi giornalieri sincronizzati per preservare la dipendenza osservata tra asset; la distribuzione futura può differire significativamente dal passato.",
        "Gli shock di cigno nero sono scenari sintetici calibrati per classe di attivo, non ricostruzioni esatte dei prezzi storici dei singoli ETF.",
        "VaR e CVaR qui riportati sono calcolati sulla distribuzione dei rendimenti terminali dell'orizzonte selezionato; non rappresentano una perdita massima garantita.",
        "La copertura put è un overlay parametrico illustrativo e non usa prezzi di opzioni, volatilità implicite, strike o scadenze di mercato. Non equivale al pricing di una strategia negoziabile.",
        "I risultati sono simulazioni condizionate ai parametri inseriti e non costituiscono garanzia di rendimento né raccomandazione di investimento."
    ]
    for note in notes:
        story.append(Paragraph("• " + note, styles["BodyText"])); story.append(Spacer(1, 0.15*cm))
    doc.build(story)
    return buffer.getvalue()
