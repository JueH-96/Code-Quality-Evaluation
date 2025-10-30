import sys

# Read input from stdin
X = float(sys.stdin.read().strip())

# Convert the float to a string without trailing zeros or decimal point if unnecessary
formatted_X = ('%.3f' % X).rstrip('0').rstrip('.')

# Print the result to stdout
print(formatted_X)