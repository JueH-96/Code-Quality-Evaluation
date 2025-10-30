import sys

# Read input from stdin
input = sys.stdin.read()
A, B = map(int, input.split())

# Calculate A^B + B^A
result = (A ** B) + (B ** A)

# Print the result to stdout
print(result)