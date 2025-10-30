from itertools import permutations
import math

def is_square(n):
    return int(math.isqrt(n)) ** 2 == n

def count_square_numbers(N, S):
    count = 0
    seen = set()
    for perm in permutations(S):
        if perm[0] == '0':
            continue
        num = int(''.join(perm))
        if num not in seen and is_square(num):
            count += 1
            seen.add(num)
    return count

N = int(input())
S = input()

print(count_square_numbers(N, S))