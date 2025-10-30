from heapq import heappush, heappop

N = int(input())
ABX = [list(map(int, input().split())) for _ in range(N-1)]

dp = [float('inf')] * N
dp[0] = 0
hq = [(0, 0)]
for i in range(N-1):
    a, b, x = ABX[i]
    while hq and hq[0][1] < i:
        heappop(hq)
    dp[i] = min(dp[i], hq[0][0] + b)
    heappush(hq, (dp[i] + a, i))
    heappush(hq, (dp[i] + a, x-1))

print(min(dp[-1], hq[0][0]))