import sys

# Read input from stdin
N = int(sys.stdin.read().strip())

# Concatenate N copies of the digit N
result = str(N) * N

# Print the resulting string
print(result)