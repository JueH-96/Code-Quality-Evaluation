# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()

MOD = 998244353

from collections import defaultdict

# dp[state] = number of strings T of length j such that
# lcs[i][j] = state[i] for all i from 0 to N
# state is a tuple of length N+1

dp = defaultdict(int)
dp[(0,) * (N + 1)] = 1

for j in range(1, M + 1):
    new_dp = defaultdict(int)
    for prev_state, count in dp.items():
        for c in range(26):  # 'a' to 'z'
            char = chr(ord('a') + c)
            new_state = [0] * (N + 1)
            new_state[0] = 0  # lcs[0][j] = 0
            for i in range(1, N + 1):
                if S[i-1] == char:
                    new_state[i] = prev_state[i-1] + 1
                else:
                    new_state[i] = max(prev_state[i], new_state[i-1])
            new_state = tuple(new_state)
            new_dp[new_state] = (new_dp[new_state] + count) % MOD
    dp = new_dp

# Count answers for each k
ans = [0] * (N + 1)
for state, count in dp.items():
    k = state[N]
    ans[k] = (ans[k] + count) % MOD

print(' '.join(map(str, ans)))