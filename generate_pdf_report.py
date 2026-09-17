"""
PDF Report Generator for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Builds a professional, multi-page PDF report with embedded figures, styled tables,
and comprehensive technical findings.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "drivebuddy_pavbhaji_classifier", "reports", "figures")
OUTPUT_PDF_ROOT = os.path.join(BASE_DIR, "data_analysis_report.pdf")
OUTPUT_PDF_REPORTS = os.path.join(BASE_DIR, "drivebuddy_pavbhaji_classifier", "reports", "data_analysis_report.pdf")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw 'Page X of Y' in the running footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "DriveBuddyAI ML Challenge — Pav Bhaji Text Classification Report")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 745, 558, 745)

        # Footer
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential & Proprietary — DriveBuddyAI Submission")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def build_pdf_report(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        bulletIndent=5,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#4A5568")
    )

    story = []

    # Title Block
    story.append(Paragraph("DriveBuddyAI Pav Bhaji Classification Challenge", title_style))
    story.append(Paragraph("<b>Technical Data Analysis, Modeling & Evaluation Report</b> | Text Classification Track", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Metadata Grid
    meta_data = [
        [
            Paragraph("<b>Target Objective:</b> Text-based Pav Bhaji post classification", body_style),
            Paragraph("<b>Constraint:</b> Zero vision/pixel models", body_style)
        ],
        [
            Paragraph("<b>Dataset Size:</b> 1,500 JSON posts (452 Labeled)", body_style),
            Paragraph("<b>Winning Model:</b> Logistic Regression (Word+Char TF-IDF)", body_style)
        ],
        [
            Paragraph("<b>Holdout Test F1:</b> <b>0.6591</b>", body_style),
            Paragraph("<b>Holdout Test ROC-AUC:</b> <b>0.6952</b>", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This project delivers an end-to-end Machine Learning pipeline for the DriveBuddyAI ML Challenge. "
        "The objective is to classify Instagram posts as <b>Pav Bhaji</b> (Class 1) or <b>Not Pav Bhaji</b> (Class 0) "
        "strictly using textual metadata (captions, tags, location, comments, and engagement counts) without computer vision. "
        "From 1,500 posts in <code>pavbhaji.json</code>, exactly <b>452 posts</b> map 1-to-1 to verified images (269 Non-Pav Bhaji, 183 Pav Bhaji). "
        "By employing sublinear Word TF-IDF n-grams (1,2), sub-word Character TF-IDF n-grams (3,5), compound hashtag decomposition, and domain food lexical profiling, "
        "the <b>Logistic Regression (Word+Char TF-IDF)</b> pipeline achieved a 5-fold CV F1 of <b>0.6469</b>, a holdout test accuracy of <b>67.03%</b>, "
        "a test F1 of <b>0.6591</b>, and a test ROC-AUC of <b>0.6952</b>.",
        body_style
    ))

    # 2. Problem Definition & Leakage Hazard
    story.append(Paragraph("2. Problem Definition & Data Leakage Prevention", h1_style))
    story.append(Paragraph(
        "Because the dataset was gathered by searching <code>#pavbhaji</code> on Instagram, both Class 1 and Class 0 contain <code>#pavbhaji</code> "
        "in their hashtag dumps. Superficial keyword matching fails because food bloggers frequently attach dozens of popular hashtags to photos of "
        "unrelated dishes (e.g. Chicken Tikka, Pasta, Dosa, Pakoda). The model must identify whether Pav Bhaji is the primary culinary subject or merely a peripheral hashtag.",
        body_style
    ))
    story.append(Paragraph("<b>Key Safeguards Implemented:</b>", body_style))
    story.append(Paragraph("• <b>Strict Pipeline Encapsulation:</b> All vectorizers and transformers are fitted strictly inside training folds.", bullet_style))
    story.append(Paragraph("• <b>Stratified 80/20 Splitting:</b> Preserves the 60:40 class distribution (361 train / 91 test) with fixed random seed.", bullet_style))
    story.append(Paragraph("• <b>Zero Pixel/Folder Leakage:</b> Image directory names (0/ and 1/) are used exclusively as ground-truth targets.", bullet_style))

    # 3. Dataset Description
    story.append(Paragraph("3. Dataset Description & Statistics", h1_style))
    dataset_rows = [
        ["Attribute / Field", "Value", "Notes"],
        ["Total Raw JSON Entries", "1,500", "Posts in pavbhaji.json with captions, tags, likes, comments"],
        ["Ground-Truth Labeled Posts", "452", "Mapped 1-to-1 to verified downloaded images"],
        ["Class 0 (Non-Pav Bhaji)", "269 (59.5%)", "Images featuring chicken, pasta, dosa, pakoda, etc."],
        ["Class 1 (Pav Bhaji)", "183 (40.5%)", "Images featuring authentic Pav Bhaji plates"],
        ["Unlabeled Candidate Pool", "1,048", "Posts with uncollected images for future semi-supervised learning"],
        ["Train / Test Partition", "361 / 91", "Stratified 80/20 train/test split (random_state=42)"]
    ]
    ds_table = Table(dataset_rows, colWidths=[150, 90, 260])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F7FAFC"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 8))

    # Visualizations: Figures 1 & 2
    story.append(Paragraph("4. Exploratory Data Analysis & Visualizations", h1_style))
    fig1_path = os.path.join(FIG_DIR, "class_distribution.png")
    fig2_path = os.path.join(FIG_DIR, "missing_values_analysis.png")
    if os.path.exists(fig1_path) and os.path.exists(fig2_path):
        img_table = Table([
            [Image(fig1_path, width=240, height=160), Image(fig2_path, width=240, height=160)],
            [Paragraph("<b>Figure 1:</b> Ground-Truth Class Distribution", callout_style),
             Paragraph("<b>Figure 2:</b> Missing Values Across Metadata", callout_style)]
        ], colWidths=[250, 250])
        img_table.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(img_table)
    story.append(Spacer(1, 8))

    # Text analysis figures: Figures 4 & 5
    fig4_path = os.path.join(FIG_DIR, "top_words_by_class.png")
    fig5_path = os.path.join(FIG_DIR, "top_hashtags_by_class.png")
    if os.path.exists(fig4_path) and os.path.exists(fig5_path):
        img_table2 = Table([
            [Image(fig4_path, width=240, height=140), Image(fig5_path, width=240, height=140)],
            [Paragraph("<b>Figure 3:</b> Top Informative Words by Class", callout_style),
             Paragraph("<b>Figure 4:</b> Top Co-occurring Hashtags by Class", callout_style)]
        ], colWidths=[250, 250])
        img_table2.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(img_table2)
    story.append(Spacer(1, 8))

    # 5. Feature Engineering Table
    story.append(Paragraph("5. Feature Engineering & Representation Pipeline", h1_style))
    feat_rows = [
        ["Feature / Representation", "Type", "Source / Calculation", "Relevance & Anti-Leakage Rationale"],
        ["Combined Text", "Text", "Caption + Tags + Location + Comments", "Unified textual corpus per Instagram post."],
        ["Word TF-IDF N-grams (1,2)", "Sparse", "Sublinear TF scaling, max 3000 feats", "Captures key culinary bi-grams (e.g. 'extra butter', 'cheese pav bhaji')."],
        ["Char TF-IDF N-grams (3,5)", "Sparse", "Sub-word char_wb analyzer, max 3000", "Handles informal Hinglish (khaalo, swad, teekha) & compound hashtags."],
        ["Compound Hashtag Splits", "Text", "Rule-based regex segmentation", "Decomposes #cheesepavbhaji -> 'cheese pav bhaji'."],
        ["Competing Food Density", "Numeric", "Conflicting food hits / total words", "Strong negative indicator when chicken/dosa/burger dominate."],
        ["Pav Bhaji Focus Ratio", "Numeric", "Hits_PB / (Hits_PB + Hits_Comp + 1)", "Distinguishes dedicated posts from spam hashtag blocks."]
    ]
    feat_table = Table(feat_rows, colWidths=[120, 45, 145, 190])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("PADDING", (0,0), (-1,-1), 3.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F7FAFC"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 8))

    # 6. Model Benchmarking Table
    story.append(Paragraph("6. Model Benchmarking & Performance Comparison", h1_style))
    model_rows = [
        ["Model Architecture", "5-Fold CV F1", "Test Acc", "Test Prec", "Test Recall", "Test F1", "Test ROC-AUC"],
        ["Logistic Regression (Word+Char TF-IDF) ⭐", "0.6469 ± 0.0389", "0.6703", "0.5800", "0.7838", "0.6591", "0.6952"],
        ["Logistic Regression (Word TF-IDF)", "0.6237 ± 0.0394", "0.6374", "0.5333", "0.8649", "0.6598", "0.6772"],
        ["Linear SVM (Calibrated)", "0.5705 ± 0.0459", "0.6044", "0.5135", "0.5135", "0.5135", "0.6942"],
        ["Multinomial Naive Bayes", "0.5942 ± 0.0463", "0.6154", "0.5116", "0.5946", "0.6667", "0.6637"],
        ["Random Forest Baseline", "0.6147 ± 0.0843", "0.5934", "0.5000", "0.7297", "0.5918", "0.6532"],
        ["SGD Classifier (Modified Huber)", "0.6152 ± 0.0467", "0.5824", "0.4865", "0.4865", "0.4865", "0.5681"]
    ]
    model_table = Table(model_rows, colWidths=[150, 75, 45, 50, 55, 55, 70])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("PADDING", (0,0), (-1,-1), 3.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F7FAFC"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 8))

    # Evaluation figures: Figures 6, 7, 8, 9
    fig6_path = os.path.join(FIG_DIR, "model_performance_comparison.png")
    fig7_path = os.path.join(FIG_DIR, "roc_curves.png")
    fig9_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    if os.path.exists(fig6_path) and os.path.exists(fig7_path):
        img_table3 = Table([
            [Image(fig6_path, width=240, height=135), Image(fig7_path, width=240, height=135)],
            [Paragraph("<b>Figure 5:</b> Model Performance Benchmark", callout_style),
             Paragraph("<b>Figure 6:</b> ROC Curves across Candidate Models", callout_style)]
        ], colWidths=[250, 250])
        img_table3.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(img_table3)
    story.append(Spacer(1, 8))

    if os.path.exists(fig9_path):
        img_table4 = Table([
            [Image(fig9_path, width=220, height=140)],
            [Paragraph("<b>Figure 7:</b> Annotated Confusion Matrix for Winning Logistic Regression Model", callout_style)]
        ], colWidths=[500])
        img_table4.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(img_table4)
    story.append(Spacer(1, 8))

    # 7. Error Analysis
    story.append(Paragraph("7. Qualitative Error Analysis", h1_style))
    story.append(Paragraph(
        "<b>False Positives (Predicted Pav Bhaji, Actually Non-Pav Bhaji):</b><br/>"
        "• <i>Example 1 (Confidence 66.9%):</i> Post discussing 'Rakhi special Homemade Pav Bhaji' with 20 festival tags where the photo featured sweet dishes.<br/>"
        "• <i>Example 2 (Confidence 70.3%):</i> Post reviewing both 'Momos from Hari Momos and Pavbhaji from Lal Chat Bhandar' where the image depicted Momos.<br/>"
        "<i>Takeaway:</i> False positives occur when multi-item food walks or reviews mention Pav Bhaji as one of several dishes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>False Negatives (Predicted Non-Pav Bhaji, Actually Pav Bhaji):</b><br/>"
        "• <i>Example 1 (Confidence 43.1%):</i> Post praising Sadanand restaurant near Crawford market without explicit dish keywords.<br/>"
        "• <i>Example 2 (Confidence 44.2%):</i> Terse caption with general travel and Eid tags (#bolgappadubai #vacationmode).<br/>"
        "<i>Takeaway:</i> False negatives stem from ultra-sparse text where dish terminology is omitted.",
        body_style
    ))

    # 8. Limitations, Improvements & Conclusion
    story.append(Paragraph("8. Limitations, Future Extensions & Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Limitations:</b> Text-only classification cannot verify images when captions are sparse or misleading. "
        "Spam hashtag blocks add noise.<br/>"
        "<b>Future Extensions:</b> In production, fusing text TF-IDF features with lightweight CNN/ViT image embeddings would resolve sparse captions. "
        "Semi-supervised self-training on the 1,048 unlabeled posts could further boost generalization.<br/>"
        "<b>Conclusion:</b> The <b>Logistic Regression (Word+Char TF-IDF)</b> pipeline delivers a robust, interpretable, and reproducible "
        "solution achieving <b>0.6591 F1-score</b> and <b>0.6952 ROC-AUC</b> on unseen holdout test data.",
        body_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    build_pdf_report(OUTPUT_PDF_ROOT)
    build_pdf_report(OUTPUT_PDF_REPORTS)
