import sys
from math import gcd
from itertools import zip_longest

N, M = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

G = gcd(N, M)
S = S[::N//G]
T = T[::M//G]

if S == T:
    print('Yes')
else:
    print('No')