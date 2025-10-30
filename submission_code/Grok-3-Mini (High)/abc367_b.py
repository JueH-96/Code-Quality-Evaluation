# YOUR CODE HERE
import sys
data = sys.stdin.read().strip()
X = data  # Read the input string
parts = X.split('.')
integer_part = parts[0]
fractional_part = parts[1]
frac_stripped = fractional_part.rstrip('0')
if not frac_stripped:
    print(integer_part)
else:
    print(integer_part + '.' + frac_stripped)