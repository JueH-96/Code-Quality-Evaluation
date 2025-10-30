T = input().strip()
N = int(input())

bags = []
for _ in range(N):
    parts = input().split()
    A_i = int(parts[0])
    strings = parts[1:]
    bags.append(strings)

len_T = len(T)
INF = float('inf')

# Initialize DP: dp[j] is the minimum cost to reach j characters
dp = [INF] * (len_T + 1)
dp[0] = 0

for bag in bags:
    new_dp = dp.copy()  # Start with the current state (not taking anything from the bag)
    for j in range(len(dp)):
        if dp[j] == INF:
            continue
        for s in bag:
            s_len = len(s)
            if j + s_len > len_T:
                continue
            if T[j:j + s_len] == s:
                if new_dp[j + s_len] > dp[j] + 1:
                    new_dp[j + s_len] = dp[j] + 1
    dp = new_dp

if dp[len_T] == INF:
    print(-1)
else:
    print(dp[len_T])