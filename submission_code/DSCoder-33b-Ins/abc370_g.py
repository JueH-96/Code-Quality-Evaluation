# YOUR CODE HERE

import sys

def solve(n, m):
    MOD = 998244353
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        if sum(divisors(i)) % 3 == 0:
            dp[i] = pow(2, i - 1, MOD)
    for _ in range(m - 1):
        new_dp = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                new_dp[i] = (new_dp[i] + dp[j] * dp[i // j]) % MOD
        dp = new_dp
    return dp[n]

def divisors(n):
    i = 1
    result = []
    while i * i <= n:
        if n % i == 0:
            result.append(i)
            if n // i != i:
                result.append(n // i)
        i += 1
    return result

def main():
    n, m = map(int, sys.stdin.readline().split())
    print(solve(n, m))

if __name__ == '__main__':
    main()