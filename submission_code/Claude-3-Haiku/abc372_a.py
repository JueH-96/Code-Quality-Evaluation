import sys

# Read input from stdin
S = input()

# Remove all '.' characters from the string
result = ''.join(c for c in S if c != '.')

# Print the result to stdout
print(result)