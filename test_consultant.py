import xgboost  # CRITICAL: Must be imported first to avoid OpenMP conflict on macOS with PyTorch
from pathlib import Path
from src.consultant.report_generator import ReportGenerator


def main():
    print("=" * 60)
    # Initialize the generator using the pulled llama3.2:latest model
    generator = ReportGenerator(ollama_model="llama3.2:latest")

    title = "AI-powered fitness application with personalized workouts, real-time workout tracking, and AI coaching"
    category = "Apps"
    main_category = "Technology"
    country = "US"
    currency = "USD"
    goal = 15000.0
    duration = 30

    print("Generating Kickstarter Campaign Intelligence Report for:")
    print(f"Title:  {title}")
    print(f"Goal:   {currency} {goal}")
    print(f"Market: {category} in {country}")
    print("-" * 60)

    report = generator.generate_report(
        title=title,
        category=category,
        main_category=main_category,
        country=country,
        currency=currency,
        goal=goal,
        duration=duration,
    )

    print("\n" + "=" * 60)
    print("GENERATED REPORT:")
    print("=" * 60)
    print(report)

    # Save the report to artifacts
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "campaign_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport successfully saved to: {report_file}")


if __name__ == "__main__":
    main()
