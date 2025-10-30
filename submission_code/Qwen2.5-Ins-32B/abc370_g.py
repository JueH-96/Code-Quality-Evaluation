import sys
from collections import defaultdict

def is_good_integer(n):
    divisors_sum = sum([i for i in range(1, n + 1) if n % i == 0])
    return divisors_sum % 3 == 0

def count_good_sequences(N, M):
    MOD = 998244353
    dp = [0] * (N + 1)
    dp[1] = 1
    for i in range(2, N + 1):
        if is_good_integer(i):
            dp[i] = pow(2, M, MOD)
        else:
            dp[i] = dp[i - 1]
    return dp[N]

def main():
    N, M = map(int, sys.stdin.readline().split())
    print(count_good_sequences(N, M))

if __name__ == "__main__":
    main()