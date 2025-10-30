import itertools
import math

def is_square(n):
    return math.isqrt(n)**2 == n

def solve():
    N = int(input().strip())
    S = input().strip()
    S = sorted(S)
    ans = 0
    for p in itertools.permutations(S):
        if p[0] == '0':
            continue
        num = int(''.join(p))
        if is_square(num):
            ans += 1
    print(ans)

solve()