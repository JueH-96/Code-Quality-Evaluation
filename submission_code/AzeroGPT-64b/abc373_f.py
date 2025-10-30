from heapq import heappop, heappush

n, w = map(int, input().split())
data = [list(map(int, input().split())) for _ in range(n)]
dp = [[0] * (w + 1) for _ in range(n + 1)]

for i, (wi, vi) in enumerate(data):
    for j in range(w, 0, -1):
        if j >= wi:
            dp[i + 1][j] = max(dp[i][j], dp[i][j - wi] + vi - 1)
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])
        else:
            dp[i + 1][j] = dp[i][j]

q = [0]
for i in range(n):
    q_next = []
    while q:
        value = heappop(q)
        for j in range(w // data[i][0] + 1):
            # Calculate the number of items and the cost of items up to i-th to put j items into the knapsack after the i-th
            temp = (vi * j - j ** 2) + value
            if temp not in q_next:
                heappush(q_next, temp)
                heappush(q_next, temp - 1)
    q = q_next

print(max(q))