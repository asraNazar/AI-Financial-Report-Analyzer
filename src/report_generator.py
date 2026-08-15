from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from config import REPORTS_DIR
from src.logger import logger


def generate_financial_report(results: dict, ai_insights: str) -> Path:
    """
    Generate a PDF financial report from analysis results and AI insights.

    Args:
        results: Financial analysis results.
        ai_insights: AI-generated financial insights.

    Returns:
        Path to the generated PDF report.
    """

    logger.info("Starting financial PDF report generation")

    report_path = REPORTS_DIR / "financial_report.pdf"

    try:
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = styles["Title"]
        heading_style = styles["Heading2"]
        body_style = styles["BodyText"]

        story = []

        # Title
        story.append(
            Paragraph(
                "Financial Transaction Analysis Report",
                title_style,
            )
        )

        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(
            Paragraph("Executive Summary", heading_style)
        )

        summary_data = [
            ["Metric", "Value"],
            ["Total Transactions", str(results["total_transactions"])],
            ["Total Amount", str(results["total_amount"])],
            ["Total Deposits", str(results["total_deposits"])],
            ["Total Withdrawals", str(results["total_withdrawals"])],
            ["Highest Transaction", str(results["highest_transactions"])],
            ["Lowest Transaction", str(results["lowest_transactions"])],
            ["Most Active Branch", str(results["most_active_branch"])],
            ["Least Active Branch", str(results["least_active_branch"])],
        ]

        summary_table = Table(
            summary_data,
            colWidths=[3 * inch, 2.5 * inch],
        )

        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Branch Performance
        story.append(
            Paragraph("Branch Performance", heading_style)
        )

        branch_summary = results["branch_summary"]

        branch_data = [["Branch", "Transaction Amount"]]

        for branch, amount in branch_summary.items():
            branch_data.append([str(branch), str(amount)])

        branch_table = Table(
            branch_data,
            colWidths=[3 * inch, 2.5 * inch],
        )

        branch_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(branch_table)
        story.append(Spacer(1, 20))

        # AI Insights
        story.append(
            Paragraph("AI Financial Insights", heading_style)
        )

        story.append(
            Paragraph(
                str(ai_insights).replace("\n", "<br/>"),
                body_style,
            )
        )

        story.append(Spacer(1, 20))

        # Charts
        branch_chart = Path("charts/branch_summary.png")
        transaction_chart = Path("charts/transaction_types.png")

        if branch_chart.exists():
            story.append(
                Paragraph("Branch Summary Chart", heading_style)
            )

            story.append(
                Image(
                    str(branch_chart),
                    width=6 * inch,
                    height=3.75 * inch,
                )
            )

            story.append(Spacer(1, 15))

        if transaction_chart.exists():
            story.append(
                Paragraph(
                    "Transaction Type Distribution",
                    heading_style,
                )
            )

            story.append(
                Image(
                    str(transaction_chart),
                    width=5 * inch,
                    height=5 * inch,
                )
            )

        doc.build(story)

        logger.info(
            f"Financial report saved: {report_path}"
        )

        return report_path

    except Exception as error:
        logger.error(
            f"Error generating financial report: {error}"
        )
        raise