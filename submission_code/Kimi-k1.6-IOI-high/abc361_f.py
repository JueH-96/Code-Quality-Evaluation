import sys
from functools import lru_cache

def integer_nth_root(n, k):
    if k == 0:
        return n
    if n == 0:
        return 0
    low = 0
    high = n
    while low < high:
        mid = (low + high + 1) // 2
        p = 1
        overflow = False
        for _ in range(k):
            p *= mid
            if p > n:
                overflow = True
                break
        if overflow or p > n:
            high = mid - 1
        else:
            low = mid
    return low

@lru_cache(maxsize=None)
def count_perfect_powers(N):
    if N < 1:
        return 0
    result = 1  # for x=1 if N >= 1
    e = 2
    while True:
        M = integer_nth_root(N, e)
        if M < 2:
            break
        result += (M - count_perfect_powers(M))
        e += 1
    return result

N = int(sys.stdin.readline())
print(count_perfect_powers(N))