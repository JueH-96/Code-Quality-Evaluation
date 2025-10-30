# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Change the last character of S to 4
modified_S = S[:-1] + '4'

# Print the modified string
print(modified_S)