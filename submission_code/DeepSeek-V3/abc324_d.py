from itertools import permutations
from math import isqrt

def count_square_numbers(N, S):
    # Generate all unique permutations of the digits
    unique_digits = list(S)
    # To handle leading zeros, we need to consider all permutations
    # but count only unique numbers
    unique_numbers = set()
    for p in permutations(unique_digits):
        num_str = ''.join(p)
        num = int(num_str)
        unique_numbers.add(num)
    # Now, count how many of these numbers are perfect squares
    count = 0
    for num in unique_numbers:
        sqrt_num = isqrt(num)
        if sqrt_num * sqrt_num == num:
            count += 1
    return count

# Read input
N = int(input())
S = input().strip()

# Compute and print the result
print(count_square_numbers(N, S))