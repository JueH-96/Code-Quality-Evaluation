import math, sys

def solve():
    import sys, math
    from sys import stdin
    N = int(stdin.readline())
    S = stdin.readline().strip()
    s_count = [0]*10
    for c in S:
        s_count[int(c)] +=1
    s_count = tuple(s_count)
    max_square = 10**N -1
    limit = math.isqrt(max_square) +1
    count=0
    for i in range(limit):
        x = i*i
        d_count = [0]*10
        tmp = x
        for _ in range(N):
            d_count[tmp %10] +=1
            tmp = tmp //10
        d_count = tuple(d_count)
        if d_count == s_count:
            count +=1
    print(count)