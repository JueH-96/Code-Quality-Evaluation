from itertools import permutations
from math import sqrt

def is_square(n):
    return sqrt(n).is_integer()

def solve():
    N = int(input())
    S = input()
    perms = set(permutations(S))
    count = 0
    for perm in perms:
        num = int(''.join(perm))
        if is_square(num):
            count += 1
    print(count)

solve()