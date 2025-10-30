# YOUR CODE HERE
import sys

# Read input from stdin
X = float(sys.stdin.read().strip())

# Format the number according to the problem requirements
formatted_X = "{:.3f}".format(X).rstrip('0').rstrip('.')

# Print the formatted number
print(formatted_X)