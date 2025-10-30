# YOUR CODE HERE
from itertools import product
from math import gcd
from functools import reduce

def modinv(a, m=998244353):
    return pow(a, m-2, m)

def lcm(a, b):
    return a * b // gcd(a, b)

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    total = 1
    for x in a:
        total = total * x % mod
    total = modinv(total)
    
    dp = [0] * (10 + 1)
    dp[0] = 1
    for x in a:
        for i in range(10, 0, -1):
            dp[i] = (dp[i] + dp[i - 1] * x) % mod
    
    ans = 0
    for i in range(n):
        for comb in product([0, 1], repeat=i):
            s = 0
            for j in range(i):
                if comb[j]:
                    s += a[j]
            if s <= 10:
                dp[10 - s] = (dp[10 - s] - 1 + mod) % mod
    
    for i in range(1, 11):
        ans = (ans + dp[i] * total) % mod
    
    print(ans)

solve()