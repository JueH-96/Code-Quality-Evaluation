from itertools import permutations
from math import sqrt

def is_square(n):
    return int(sqrt(n)) ** 2 == n

def count_square_numbers(S):
    N = len(S)
    digits = [int(d) for d in S]
    square_count = 0

    for perm in permutations(digits):
        num = sum(perm[i] * 10 ** (N - i - 1) for i in range(N))
        if is_square(num):
            square_count += 1

    return square_count

# Read input
N = int(input())
S = input()

# Solve and print the answer
print(count_square_numbers(S))