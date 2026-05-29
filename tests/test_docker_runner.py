"""Test Docker sandbox runner functionality."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.runner import DockerRunner

# Simple test script
test_script = """
print("Hello from Docker sandbox!")
x = 10
y = 20
print(f"x + y = {x + y}")
"""

try:
    runner = DockerRunner()
    print(f"Docker runner initialized: {runner}")
    
    # Check if image exists
    if runner.image_exists():
        print(f"Sandbox image '{runner.image}' found")
    else:
        print(f"Sandbox image '{runner.image}' not found")
    
    # Run test script
    print("\nRunning test script...")
    output = runner.run(test_script)
    print(f" Script executed successfully")
    print(f"Output:\n{output}")
    
except Exception as e:
    print(f" Error: {e}")