import sys
from collections import Counter

def solve():
    mod = 998244353
    N, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    T1 = T[0]
    cnt = Counter(T)
    inv_N = pow(N, mod - 2, mod)
    dp = [0] * (X +1)
    dp[0] =1
    unique_T = list(cnt.keys())
    for s in range(0, X+1):
        current = dp[s]
        if current ==0:
            continue
        for t in unique_T:
            s_new = s + t
            if s_new > X:
                continue
            dp[s_new] = (dp[s_new] + current * cnt[t] * inv_N) % mod
    a = max(0, X +1 - T1)
    total = 0
    # To speed up the sum, iterate and add
    for s in range(a, X +1):
        total = (total + dp[s]) % mod
    prob = total * inv_N % mod
    print(prob)