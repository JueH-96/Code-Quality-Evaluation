import sys
from math import gcd

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    p = 1
    for i in range(n):
        p = (p * a[i]) % k
    if p == 0:
        print(0)
    else:
        print(k - p)

t = int(input())
for _ in range(t):
    solve()