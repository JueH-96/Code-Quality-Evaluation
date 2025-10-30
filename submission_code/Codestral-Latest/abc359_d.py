MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    q = S.count('?')
    dp = [[[0] * 2 for _ in range(K)] for _ in range(N + 1)]
    dp[0][0][0] = 1

    for i in range(N):
        for j in range(K):
            for k in range(2):
                if S[i] == '?':
                    for c in range(2):
                        if j == 0 or (j > 0 and dp[i][j - 1][k] != 0 and (k != c or j == K - 1)):
                            dp[i + 1][(j + 1) % K][c] = (dp[i + 1][(j + 1) % K][c] + dp[i][j][k]) % MOD
                else:
                    c = 0 if S[i] == 'A' else 1
                    if j == 0 or (j > 0 and dp[i][j - 1][k] != 0 and (k != c or j == K - 1)):
                        dp[i + 1][(j + 1) % K][c] = (dp[i + 1][(j + 1) % K][c] + dp[i][j][k]) % MOD

    result = 0
    for j in range(K):
        for k in range(2):
            result = (result + dp[N][j][k]) % MOD

    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
S = data[2]

# Calculate and print the result
print(count_good_strings(N, K, S))