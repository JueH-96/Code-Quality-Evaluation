import sys

# Read the input as a string
x_str = sys.stdin.readline().strip()

# Remove trailing zeros
# Start from the end and remove '0' as long as it's the last character
while x_str.endswith('0'):
    x_str = x_str[:-1]

# After removing trailing zeros, check if the string ends with a decimal point
# This happens if the original number was an integer like 12.000 which became "12."
if x_str.endswith('.'):
    x_str = x_str[:-1]

# Print the result
print(x_str)