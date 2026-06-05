import os
import sys
import time
import subprocess
from datetime import datetime

# ─────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────
PIPELINE_NAME = "Silent Churn MLOps Pipeline"
VERSION = "1.0.0"

def print_header():
    print("=" * 60)
    print(f"  {PIPELINE_NAME}")
    print(f"  Version: {VERSION}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def print_step(step_num, step_name):
    print(f"\n{'─' * 60}")
    print(f"  Step {step_num}: {step_name}")
    print(f"{'─' * 60}")

def run_step(script_path, step_name):
    print(f"\n▶ Running {step_name}...")
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    if result.returncode == 0:
        print(result.stdout)
        print(f"✅ {step_name} completed in {duration} seconds!")
        return True
    else:
        print(f"❌ {step_name} failed!")
        print(f"Error: {result.stderr}")
        return False

def run_pipeline():
    # Print header
    print_header()

    # Track results
    results = {}
    total_start = time.time()

    # ─────────────────────────────────────
    # Step 1: Feature Engineering
    # ─────────────────────────────────────
    print_step(1, "Feature Engineering")
    results['feature_engineering'] = run_step(
        'src/feature_engineering.py',
        'Feature Engineering'
    )

    if not results['feature_engineering']:
        print("❌ Pipeline stopped at Feature Engineering!")
        sys.exit(1)

    # ─────────────────────────────────────
    # Step 2: Model Training
    # ─────────────────────────────────────
    print_step(2, "Model Training")
    results['model_training'] = run_step(
        'src/train.py',
        'Model Training'
    )

    if not results['model_training']:
        print("❌ Pipeline stopped at Model Training!")
        sys.exit(1)

    # ─────────────────────────────────────
    # Step 3: Model Explanation
    # ─────────────────────────────────────
    print_step(3, "SHAP Explanation")
    results['explanation'] = run_step(
        'src/explain.py',
        'SHAP Explanation'
    )

    # ─────────────────────────────────────
    # Step 4: Monitoring
    # ─────────────────────────────────────
    print_step(4, "Monitoring")
    results['monitoring'] = run_step(
        'monitoring/monitor.py',
        'Monitoring'
    )

    # ─────────────────────────────────────
    # Step 5: Pipeline Summary
    # ─────────────────────────────────────
    total_duration = round(time.time() - total_start, 2)

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)

    for step, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {step.replace('_', ' ').title()}: {status}")

    print(f"\n  Total Duration: {total_duration} seconds")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n⚠️ Pipeline completed with some failures!")

    print("=" * 60)

# ─────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()