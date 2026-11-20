"""Build the PDF technical report from results.json and figures/."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)
from PIL import Image as PI

R = json.load(open("results.json"))
M = R["models"]; LR, DT, NN = M["Logistic Regression"], M["Decision Tree"], M["Neural Network (MLP)"]
NAVY = colors.HexColor("#1F3A5F")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.6, leading=13.5, alignment=4, spaceAfter=5)
bul = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, textColor=NAVY, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=colors.HexColor("#2F6DA3"), spaceBefore=6, spaceAfter=4)
cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.3, alignment=1, textColor=colors.HexColor("#444444"), spaceAfter=8)
ttl = ParagraphStyle("t", parent=ss["Title"], fontSize=20, textColor=NAVY, leading=25)
cen = ParagraphStyle("cen", parent=ss["Normal"], alignment=1, fontSize=11, spaceAfter=4)

def P(t): return Paragraph(t, body)
def B(t): return Paragraph(t, bul, bulletText="\u2022")
def fig(path, caption, width=15.5*cm):
    w, h = PI.open(path).size
    return KeepTogether([Image(path, width=width, height=width*h/w), Paragraph(caption, cap)])
def table(data, widths):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8.8), ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EAF0F7")]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    return t
def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2*cm, 1.2*cm, "Akshita Goel | Roll No. 2301730231 | ENSP355 - AI in Cyber Security Lab")
    c.drawRightString(A4[0]-2*cm, 1.2*cm, f"Page {d.page}"); c.restoreState()

bc, bp = R["by_country"], R["by_protocol"]
cl = R["corr_label"]; th = {r["t"]: r for r in R["threshold"]}
fc, fb = R["fpr_clean"], R["fpr_biased"]
fk = "Foreign (US/UK/DE/FR)"
s = []
s += [Spacer(1, 1*cm), Paragraph("K.R. Mangalam University", cen),
      Paragraph("AI-Driven Cyber Threat Awareness<br/>and Detection Prototype", ttl),
      Paragraph("Lab Assignment 01 (Unit 1) - Technical Report", cen), Spacer(1, 0.4*cm)]
info = [["Student Name", "Akshita Goel"], ["Roll Number", "2301730231"],
        ["Program / Section", "B.Tech CSE (AI & ML) - Section D"],
        ["Course", "AI in Cyber Security Lab (ENSP355)"], ["Submitted to", "Mrs. Monika Khatikar"]]
t = Table(info, colWidths=[4.5*cm, 8*cm], hAlign="CENTER")
t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.grey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF0F7")), ("FONTSIZE", (0,0), (-1,-1), 9.5)]))
s += [t, Paragraph("1. Problem Overview", h1)]
s += [P("Organisations rely on digital infrastructure and face phishing, malware, insider misuse and automated attacks. "
        "Fixed signature rules struggle with adaptive threats, whereas machine learning can learn patterns from historical events and flag unusual behaviour. "
        "This project plays the role of an entry-level security analyst who builds a small AI-enabled prototype that (a) studies the threat landscape, "
        "(b) classifies each event as <b>normal</b> or <b>suspicious</b> using classical ML and a neural network, and (c) examines the ethical and operational risks of automating such decisions.")]
s += [Paragraph("Dataset", h2),
      P(f"A synthetic dataset of <b>{R['n']}</b> login / network-summary events was generated with a fixed seed (231). It has 11 features: protocol, source country, device type, "
        f"login hour, failed logins, session duration, bytes in, bytes out, ports accessed, password age and an MFA flag. <b>{R['n_susp']} events ({R['susp_pct']}%)</b> are labelled suspicious. "
        "Labels come from a risk score built from well-known indicators (high-risk geolocation, remote-access protocols, repeated failed logins, night-time logins, large outbound transfer, "
        "many ports, no MFA, old passwords) plus random noise, so the task is learnable but not trivial. About 3% of the values in four columns were removed on purpose to practise preprocessing. No real user data is used.")]
s += [Paragraph("2. Methodology", h1)]
for x in ["<b>Environment (Task 1):</b> Python virtual environment with numpy, pandas, scikit-learn, matplotlib, seaborn and jupyter; checked with <i>setup_check.py</i>. Repository: README.md, .gitignore, requirements.txt, data/, src/.",
          "<b>EDA (Task 2):</b> class balance, suspicious rate by country and protocol, distributions of failed logins, login hour, password age, bytes out, ports and session length, and a correlation heatmap.",
          "<b>Preprocessing (Task 3):</b> median imputation and standard scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical columns; 75/25 stratified train-test split, all wrapped in a scikit-learn Pipeline so that no test information leaks into training.",
          "<b>Models:</b> Logistic Regression and a Decision Tree (max depth 5) as ML models; scikit-learn MLPClassifier (32-16 hidden units, early stopping) as the deep-learning prototype.",
          "<b>Metrics:</b> accuracy, precision, recall, F1, confusion matrix and training time on the held-out test set.",
          "<b>Ethics (Task 5):</b> decision-threshold sweep, a biased-label simulation with per-group false-positive rates, and an explainability comparison."]:
    s.append(B(x))
s.append(PageBreak())

s += [Paragraph("3. Results and Observations", h1), Paragraph("3.1 Threat Landscape Analysis", h2)]
for x in [f"Class balance: only {R['susp_pct']}% of events are suspicious, so accuracy alone can be misleading (predicting 'normal' for everything would already score about 82%).",
          f"Source country is the strongest indicator: RU {bc['RU']}%, UNKNOWN {bc['UNKNOWN']}% and CN {bc['CN']}% of events are suspicious, against roughly 9-11% for IN, US, UK, DE and FR.",
          f"Remote-access protocols stand out: RDP {bp['RDP']}%, FTP {bp['FTP']}% and SSH {bp['SSH']}% suspicious, versus about 12-14% for HTTPS, DNS and HTTP.",
          f"Correlation with the label is highest for MFA disabled (r = {cl['mfa_enabled']}), ports accessed (r = {cl['ports_accessed']}) and failed logins (r = {cl['failed_logins']}); outbound bytes follow (r = {cl['bytes_out']}). Login hour and session length show almost no linear relationship."]:
    s.append(B(x))
s += [fig("figures/fig1_class_balance.png", "Figure 1: Normal vs suspicious events", 8*cm),
      fig("figures/fig2_rate_country_protocol.png", "Figure 2: Suspicious rate by country and protocol"),
      fig("figures/fig3_distributions.png", "Figure 3: Failed logins, login hour and password age by class"),
      fig("figures/fig4_boxplots.png", "Figure 4: Attack-related feature distributions (bytes_out on log scale)"),
      fig("figures/fig5_heatmap.png", "Figure 5: Correlation heatmap", 11*cm)]

s += [Paragraph("3.2 ML Threat Classification (Task 3)", h2)]
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "Train time (s)"]]
for n, m in M.items(): rows.append([n, m["accuracy"], m["precision"], m["recall"], m["f1"], m["train_s"]])
s += [table(rows, [4.6*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.8*cm]), Spacer(1, 6),
      P(f"Logistic Regression reaches {LR['accuracy']*100:.1f}% accuracy but its recall is only {LR['recall']*100:.1f}%, so it misses more than half of the attacks: "
        f"{LR['cm'][1][0]} of {R['test_susp']} suspicious test events were passed as normal. The Decision Tree is slightly worse here (F1 {DT['f1']}). "
        "Because the threat signal is a combination of several moderate indicators plus noise, a single linear boundary or a shallow tree cannot capture it fully. "
        "In security, recall and F1 matter more than accuracy because a missed attack is costly and the data is imbalanced."),
      fig("figures/fig6_confusion.png", "Figure 6: Confusion matrices for all three models")]

ratio_lr = NN["train_s"]/LR["train_s"]; ratio_dt = NN["train_s"]/DT["train_s"]
s += [Paragraph("3.3 Deep Learning Prototype and ML vs DL (Task 4)", h2),
      P(f"The MLP gave the best detection: accuracy {NN['accuracy']*100:.1f}%, recall {NN['recall']*100:.1f}%, F1 {NN['f1']}. It caught {NN['cm'][1][1]} attacks compared with {LR['cm'][1][1]} for Logistic Regression "
        f"and {DT['cm'][1][1]} for the tree. However, it needed {NN['train_s']} s to train versus {LR['train_s']} s (Logistic Regression, about {ratio_lr:.1f}x) and {DT['train_s']} s (Decision Tree, about {ratio_dt:.0f}x). "
        f"Prediction time is similar for all models (below 0.01 s on {R['test_n']} events).")]
for x in ["<b>Training time:</b> DL is the most expensive; harmless at 5,000 rows but it grows quickly with millions of events.",
          "<b>Detection accuracy:</b> the MLP gains a modest improvement by learning feature interactions automatically.",
          "<b>Practical feasibility:</b> on small tabular data a tree or linear model is cheap and easy to explain; DL becomes worthwhile for large, complex data (raw traffic, sequences) when enough compute and monitoring are available."]:
    s.append(B(x))
s.append(fig("figures/fig7_ml_vs_dl.png", "Figure 7: ML vs DL - performance and training time"))

s += [Paragraph("4. Ethical Discussion (Task 5)", h1), Paragraph("4.1 False positives and thresholds", h2)]
rows = [["Threshold", "False pos.", "False neg.", "FPR", "Recall", "Precision"]]
for r in R["threshold"]: rows.append([r["t"], r["fp"], r["fn"], f"{r['fpr']}%", f"{r['recall']}%", f"{r['precision']}%"])
s += [table(rows, [2.6*cm]*6), Spacer(1, 6),
      P(f"Lowering the threshold to 0.1 catches {th[0.1]['recall']}% of attacks but wrongly flags {th[0.1]['fp']} legitimate events ({th[0.1]['fpr']}% of normal users), which would lock out genuine users. "
        f"Raising it to 0.7 cuts false positives to {th[0.7]['fp']} but misses {th[0.7]['fn']} attacks (recall {th[0.7]['recall']}%). No threshold removes both errors, so the choice is a business and ethical decision, not only a technical one."),
      fig("figures/fig8_threshold.png", "Figure 8: Threshold trade-off", 10*cm),
      Paragraph("4.2 Biased data and unfair decisions", h2),
      P(f"To simulate biased data, {R['flipped']} legitimate foreign-origin events (US, UK, DE, FR) in the training set were wrongly relabelled as suspicious, as if analysts distrusted foreign traffic. "
        f"Tested on clean data, the false-positive rate for these foreign users rose from {fc[fk]}% to <b>{fb[fk]}%</b>, while for Indian users it stayed near {fb['India (IN)']}%. "
        "The model learned the human bias and now blocks roughly six in ten legitimate foreign users, even though overall accuracy still looks acceptable. Fairness must therefore be measured per group."),
      fig("figures/fig9_bias.png", "Figure 9: False-positive rate by group, clean vs biased training labels", 9*cm),
      Paragraph("4.3 Ethical risks, privacy and explainability", h2)]
for x in ["<b>Ethical risks:</b> false positives deny service to genuine users; biased labels create discrimination (for example by country); over-reliance on automation can remove human review; attackers can adapt to evade the model.",
          "<b>Privacy concerns:</b> detection needs logs with IPs, locations, devices and behaviour, which are personal data. Mitigations: data minimisation, anonymisation / pseudonymisation, access control, retention limits and user consent.",
          "<b>Explainability limits:</b> Logistic Regression has readable coefficients (country, MFA and protocol terms dominate). The MLP is a black box; permutation importance only approximates what it relies on. Users blocked by a black box cannot easily receive a reason or appeal.",
          "<b>Recommendations:</b> keep a human in the loop for blocking decisions, audit fairness per group, set thresholds according to the cost of each error, prefer interpretable models where possible, and monitor data drift."]:
    s.append(B(x))
s += [fig("figures/fig10_explain.png", "Figure 10: Interpretable coefficients vs approximate importance for the MLP"),
      Paragraph("5. Conclusion", h1),
      P("The prototype shows how data-driven methods help analysts spot threats. EDA surfaced the key indicators (high-risk country, remote-access protocols, missing MFA, failed logins), "
        "the MLP gave the best detection, and the simple models offered the best speed and transparency. The ethical experiments showed that thresholds trade user disruption against missed attacks "
        "and that biased training data produces unfair outcomes. AI should therefore assist, not replace, human security analysts."),
      Paragraph("References", h1)]
for x in ["Pedregosa et al., scikit-learn documentation - https://scikit-learn.org",
          "NIST Cybersecurity Framework - https://www.nist.gov/cyberframework",
          "Dataset: synthetic, generated by src/generate_dataset.py (seed 231); no external dataset used."]:
    s.append(B(x))
SimpleDocTemplate("Akshita_Goel_2301730231_Lab01_Report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                  topMargin=1.8*cm, bottomMargin=2*cm, title="AI-Driven Cyber Threat Awareness and Detection Prototype",
                  author="Akshita Goel").build(s, onFirstPage=footer, onLaterPages=footer)
print("built")
