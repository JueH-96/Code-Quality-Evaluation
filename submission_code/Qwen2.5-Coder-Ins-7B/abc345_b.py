import sys

# Read input from stdin
X = int(sys.stdin.read().strip())

# Calculate the ceiling of X/10
result = (X + 9) // 10

# Print the result
print(result)