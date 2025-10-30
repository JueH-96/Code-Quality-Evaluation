import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Remove all '.' from the string
result = S.replace('.', '')

# Print the result to stdout
print(result)