MOD = 998244353

N, K = map(int, input().split())
S = input().strip()

if K == 1:
    count = 1
    for c in S:
        if c == '?':
            count *= 2
            count %= MOD
    print(count)
    exit()

def is_palindrome(s):
    return s == s[::-1]

from collections import defaultdict

dp = [defaultdict(int) for _ in range(N+1)]
dp[0][''] = 1

for i in range(1, N+1):
    current_char = S[i-1]
    next_dp = defaultdict(int)
    if current_char == '?':
        chars = ['A', 'B']
    else:
        chars = [current_char]
    for s in dp[i-1]:
        for c in chars:
            new_s = s[1:] + c
            window = s + c
            if not is_palindrome(window):
                next_dp[new_s] = (next_dp[new_s] + dp[i-1][s]) % MOD
    dp[i] = next_dp

print(sum(dp[N].values()) % MOD)