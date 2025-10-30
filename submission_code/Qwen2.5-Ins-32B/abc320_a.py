import sys

# Read input from stdin
A, B = map(int, sys.stdin.readline().strip().split())

# Calculate A^B + B^A
result = A**B + B**A

# Write the answer to stdout
print(result)