import itertools
import math

def is_square(n):
    root = int(math.isqrt(n))
    return root * root == n

def count_square_numbers(N, S):
    # Generate all permutations of the string S
    permutations = itertools.permutations(S)

    # Set to store unique square numbers
    square_numbers = set()

    for perm in permutations:
        num = int(''.join(perm))
        if is_square(num):
            square_numbers.add(num)

    return len(square_numbers)

# Read input
N = int(input().strip())
S = input().strip()

# Calculate and print the result
result = count_square_numbers(N, S)
print(result)