import itertools
import math

def is_square(n):
    """Check if a number is a perfect square."""
    root = math.isqrt(n)
    return root * root == n

def count_squares(n, s):
    """Count the number of square numbers that can be obtained by permuting the digits of s."""
    count = 0
    # Generate all permutations of the digits
    for p in set(int(''.join(map(str, p)) for p in itertools.permutations(s))):
        # Check if the permutation is a square number
        if is_square(p):
            count += 1
    return count

# Read the input
n = int(input())
s = input()

# Count the number of square numbers
count = count_squares(n, s)

# Print the result
print(count)