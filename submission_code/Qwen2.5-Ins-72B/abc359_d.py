# YOUR CODE HERE
MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    dp = [[0] * 2 for _ in range(N + 1)]
    dp[0][0] = 1  # dp[i][0] = number of good strings of length i ending with A
    dp[0][1] = 1  # dp[i][1] = number of good strings of length i ending with B

    for i in range(1, N + 1):
        for j in range(2):
            for k in range(2):
                if S[i - 1] == '?' or (S[i - 1] == 'A' and j == 0) or (S[i - 1] == 'B' and j == 1):
                    new_str = ('A' if j == 0 else 'B') + ('A' if k == 0 else 'B')
                    if not is_palindrome(new_str[-K:]):
                        dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD

    return (dp[N][0] + dp[N][1]) % MOD

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
S = data[2]

print(count_good_strings(N, K, S))