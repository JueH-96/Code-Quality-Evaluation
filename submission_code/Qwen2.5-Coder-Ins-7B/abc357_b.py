# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Count the number of uppercase and lowercase letters
uppercase_count = sum(1 for char in S if char.isupper())
lowercase_count = sum(1 for char in S if char.islower())

# Convert the string based on the counts
if uppercase_count > lowercase_count:
    result = S.lower()
else:
    result = S.upper()

# Print the result
print(result)