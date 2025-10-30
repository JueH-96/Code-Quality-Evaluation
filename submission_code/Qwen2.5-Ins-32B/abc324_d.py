import itertools
import math

def is_square(n):
    return int(math.isqrt(n)) ** 2 == n

def count_square_numbers(N, S):
    digits = list(S)
    unique_permutations = set(itertools.permutations(digits))
    count = 0
    for perm in unique_permutations:
        num = int(''.join(perm))
        if is_square(num):
            count += 1
    return count

N = int(input())
S = input().strip()
print(count_square_numbers(N, S))