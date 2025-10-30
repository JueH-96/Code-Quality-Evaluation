# YOUR CODE HERE
N, K = map(int, input().split())
S = input().strip()

MOD = 998244353
from collections import defaultdict

dp = [defaultdict(int) for _ in range(N+1)]
dp[0][()] = 1  # Empty history

for i in range(N):
    s = S[i]

    # For current position, get all previous possible histories
    for history in dp[i]:
        # Attempt to place 'A' or 'B' depending on the character at position i
        for c in ['A', 'B']:
            if s == '?' or s == c:
                new_history = history + (c,)
                if len(new_history) > K-1:
                    new_history = new_history[1:]  # Keep only last K-1 characters

                need_check = (i >= K - 1)
                is_bad = False

                if need_check:
                    # Reconstruct last K characters to check for palindrome
                    # Last K characters are history[-(K-1):] + (c,)
                    last_K_chars = history[-(K-1):] + (c,)
                    if last_K_chars == last_K_chars[::-1]:
                        is_bad = True  # Found a palindrome of length K

                if not is_bad:
                    dp[i+1][new_history] = (dp[i+1][new_history] + dp[i][history]) % MOD

# Sum up all ways at position N
ans = sum(dp[N].values()) % MOD
print(ans)