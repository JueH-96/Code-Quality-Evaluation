# YOUR CODE HERE
T = input().strip()
N = int(input())

bags = []
for i in range(N):
    line = input().strip().split()
    A_i = int(line[0])
    strings = line[1:A_i+1]
    bags.append(strings)

# DP
INF = float('inf')
m = len(T)

# dp[i][j] = minimum cost to process first i bags and build first j characters of T
dp = [[INF] * (m + 1) for _ in range(N + 1)]
dp[0][0] = 0

for i in range(1, N + 1):
    # Option 1: Do nothing
    for j in range(m + 1):
        dp[i][j] = dp[i-1][j]
    
    # Option 2: Select a string from bag i
    for s in bags[i-1]:
        s_len = len(s)
        for j in range(m + 1):
            if dp[i-1][j] < INF and j + s_len <= m:
                if T[j:j+s_len] == s:
                    dp[i][j+s_len] = min(dp[i][j+s_len], dp[i-1][j] + 1)

if dp[N][m] == INF:
    print(-1)
else:
    print(dp[N][m])