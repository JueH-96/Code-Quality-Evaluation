import sys
from functools import reduce
from operator import mul

MOD = 998244353

def read_input():
    n, x = map(int, sys.stdin.readline().split())
    t = list(map(int, sys.stdin.readline().split()))
    return n, x, t

def solve(n, x, t):
    dp = [0] * (x + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(x, -1, -1):
            for k in range(1, min(j // t[i - 1] + 1, 100)):
                dp[j] = (dp[j] + dp[j - k * t[i - 1]]) % MOD
    return dp[x]

def main():
    n, x, t = read_input()
    print(solve(n, x, t))

if __name__ == "__main__":
    main()