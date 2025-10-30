# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Split the string by '.' and get the last part
result = S.split('.')[-1]

# Print the result
print(result)