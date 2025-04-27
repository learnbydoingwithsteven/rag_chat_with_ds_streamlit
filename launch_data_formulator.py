
# Data Formulator launcher script
import os
import sys
import subprocess

# Run Data Formulator
cmd = [sys.executable, "-m", "data_formulator", "--port", "5003"]


try:
    print("Starting Data Formulator...")
    print(f"Command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    print(f"Data Formulator started with PID: {process.pid}")
    print(f"Access at: http://localhost:5003")
except Exception as e:
    print(f"Error launching Data Formulator: {str(e)}")
