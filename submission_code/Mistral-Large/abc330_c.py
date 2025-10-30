import sys
import math

def find_min_value(D):
    # We need to find the minimum value of |x^2 + y^2 - D| for non-negative integers x and y.
    # We can do this by iterating over possible values of x and y and calculating the expression.

    min_value = float('inf')
    limit = int(math.sqrt(D)) + 1

    for x in range(limit):
        for y in range(limit):
            value = x**2 + y**2
            if value >= D:
                min_value = min(min_value, value - D)
            else:
                min_value = min(min_value, D - value)

    return min_value

# Read input from stdin
D = int(sys.stdin.read().strip())

# Find the minimum value
result = find_min_value(D)

# Write the output to stdout
print(result)