T = input().strip()
N = int(input())

bags = []
for _ in range(N):
    line = input().split()
    A = int(line[0])
    strings = line[1:A+1]
    bags.append(strings)

n = len(T)

# dp[i][j] = minimum cost to have prefix T[0:j] after processing first i bags
INF = float('inf')
dp = [[INF] * (n + 1) for _ in range(N + 1)]

# Base case: empty string with 0 cost
dp[0][0] = 0

for i in range(N):
    for j in range(n + 1):
        if dp[i][j] == INF:
            continue
        
        # Option 1: do nothing
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        
        # Option 2: select a string from bag i
        for s in bags[i]:
            if j + len(s) <= n and T[j:j+len(s)] == s:
                dp[i+1][j+len(s)] = min(dp[i+1][j+len(s)], dp[i][j] + 1)

result = dp[N][n]
print(result if result != INF else -1)