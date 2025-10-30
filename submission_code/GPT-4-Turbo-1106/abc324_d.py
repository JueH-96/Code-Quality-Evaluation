from itertools import permutations
import math

def is_square(n):
    root = math.isqrt(n)
    return n == root * root

def count_square_permutations(S):
    unique_numbers = set()
    for perm in permutations(S):
        # Skip permutations that start with '0' as they are not valid numbers
        if perm[0] == '0':
            continue
        num = int(''.join(perm))
        if is_square(num):
            unique_numbers.add(num)
    return len(unique_numbers)

# Read input
N = int(input().strip())
S = input().strip()

# Calculate and print the answer
print(count_square_permutations(S))