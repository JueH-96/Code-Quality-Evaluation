# YOUR CODE HERE
import sys

# Read input from stdin
input_value = sys.stdin.read().strip()

# Convert the input to a float and then back to a string to remove unnecessary trailing zeros
output_value = str(float(input_value))

# Print the result
print(output_value)