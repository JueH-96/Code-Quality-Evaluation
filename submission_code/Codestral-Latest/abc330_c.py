import math

def find_minimum_value(D):
    # Find the integer part of the square root of D
    sqrt_D = int(math.isqrt(D))

    # Initialize the minimum value to a large number
    min_value = float('inf')

    # Check all non-negative integers x and y up to sqrt_D
    for x in range(sqrt_D + 1):
        for y in range(sqrt_D + 1):
            value = abs(x**2 + y**2 - D)
            if value < min_value:
                min_value = value

    return min_value

# Read input
D = int(input().strip())

# Calculate and print the result
result = find_minimum_value(D)
print(result)