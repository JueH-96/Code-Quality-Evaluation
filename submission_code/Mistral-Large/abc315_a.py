import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Define the vowels to be removed
vowels = "aeiou"

# Remove vowels from the string
result = ''.join([char for char in S if char not in vowels])

# Print the resulting string
print(result)