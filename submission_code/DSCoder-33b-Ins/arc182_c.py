# YOUR CODE HERE

import sys

def count_divisors(n):
    count = 0
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            count += 2 if i != n//i else 1
    return count

def solve(N, M):
    mod = 998244353
    dp = [0]*(N+1)
    dp[0] = 1
    for i in range(1, N+1):
        dp[i] = dp[i-1]*M % mod
    ans = 0
    for i in range(1, N+1):
        ans = (ans + dp[i]*count_divisors(i)) % mod
    return ans

def main():
    N, M = map(int, sys.stdin.readline().split())
    print(solve(N, M))

if __name__ == '__main__':
    main()