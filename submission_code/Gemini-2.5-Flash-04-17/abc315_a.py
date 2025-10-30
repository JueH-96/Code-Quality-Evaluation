import sys

# Read the input string from stdin
s = sys.stdin.readline().strip()

# Define the vowels
vowels = 'aeiou'

# Create a new string by iterating through the original string
# and including only characters that are not vowels
result = ""
for char in s:
  if char not in vowels:
    result += char

# Print the resulting string to stdout
print(result)