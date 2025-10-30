from collections import defaultdict

T = input().strip()
N = int(input())

# states[i][j]: Minimum cost to reach the prefix of T up to character j using strings from bags up to i
dp = defaultdict(lambda: 10000)
dp[0][0] = 0

for i in range(1, N + 1):
    A = int(input())
    strings = input().split()[1:]
    for suffix in range(len(T) + 1):
        for string in strings:
            next_suffix = suffix + len(string)
            if next_suffix <= len(T) and T[suffix:next_suffix] == string:
                dp[i][next_suffix] = min(dp[i][next_suffix], dp[i - 1][suffix] + int(string))

min_cost = min(dp[N].values())
print(min_cost if min_cost < 10000 else -1)