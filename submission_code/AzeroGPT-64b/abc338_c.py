from math import inf

N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

lo, hi = -inf, sum(Q) // min([a for a in A if a > 0]+[b for b in B if b > 0])
while hi - lo > 1:
    mid = (lo + hi) / 2
    cA, cB = floor(mid), floor(mid)
    f = all([Q[i] >= b*cB + (mid-cB)*a for i, (a, b) in enumerate(zip(A, B))])
    if f:
        lo = mid
    else:
        hi = mid

print(floor(lo))