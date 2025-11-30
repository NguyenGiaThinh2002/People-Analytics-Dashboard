"""
Simple script to run the Streamlit dashboard.
Usage: python run_dashboard.py
"""

import subprocess
import sys
import os

def main():
    # Get the dashboard app path
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'app.py')
    
    print("=" * 50)
    print("Starting PeopleAnalytics Live Dashboard")
    print("=" * 50)
    print(f"\nDashboard file: {dashboard_path}")
    print("\nOpening in browser...")
    print("Press Ctrl+C to stop the server\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    main()

