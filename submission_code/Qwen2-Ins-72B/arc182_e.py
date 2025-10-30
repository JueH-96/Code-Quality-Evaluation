from bisect import bisect
from math import ceil

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

if C == 0:
    print(sum(A[:K]))
    return

def get_min_mods(A, M, C):
    min_mods = []
    for i, a in enumerate(A):
        if i == 0:
            min_mods.append(a % M)
        else:
            min_mod = min_mods[-1]
            if a - min_mod >= M:
                min_mods.append(min_mod)
            else:
                min_mods.append(a % M)
    return min_mods

min_mods = get_min_mods(A, M, C)

def get_cycle_length(M, C):
    seen = set()
    x = 0
    cycle_length = 0
    while x not in seen:
        seen.add(x)
        x = (x + C) % M
        cycle_length += 1
    return cycle_length

cycle_length = get_cycle_length(M, C)

def get_cycle_sum(min_mods, M, C, cycle_length):
    cycle_sum = 0
    for i in range(cycle_length):
        cycle_sum += min_mods[bisect(min_mods, (i * C) % M) - 1]
    return cycle_sum

cycle_sum = get_cycle_sum(min_mods, M, C, cycle_length)

full_cycles = K // cycle_length
remainder = K % cycle_length

def get_remainder_sum(min_mods, M, C, remainder):
    remainder_sum = 0
    for i in range(remainder):
        remainder_sum += min_mods[bisect(min_mods, (i * C) % M) - 1]
    return remainder_sum

remainder_sum = get_remainder_sum(min_mods, M, C, remainder)

print(cycle_sum * full_cycles + remainder_sum)