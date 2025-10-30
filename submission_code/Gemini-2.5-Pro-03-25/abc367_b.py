# YOUR CODE HERE
import sys
from decimal import Decimal, getcontext

# Read the input string from standard input
x_str = sys.stdin.readline().strip()

# Convert the input string to a Decimal object.
# Using Decimal ensures exact representation and avoids potential
# floating-point inaccuracies that might occur with float.
d = Decimal(x_str)

# Normalize the Decimal object.
# The normalize() method removes trailing zeros from the fractional part.
# If the fractional part becomes zero after removing trailing zeros,
# it also removes the decimal point, resulting in an integer representation.
# For example:
# Decimal('1.012').normalize() -> Decimal('1.012')
# Decimal('12.340').normalize() -> Decimal('12.34')
# Decimal('99.900').normalize() -> Decimal('99.9')
# Decimal('0.000').normalize()  -> Decimal('0')
# Decimal('5.000').normalize()  -> Decimal('5')
normalized_d = d.normalize()

# Convert the normalized Decimal object back to a string for printing.
output_str = str(normalized_d)

# Print the final result to standard output.
print(output_str)