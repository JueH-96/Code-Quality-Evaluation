import sys
import heapq
from math import sqrt

def main():
    N = int(input())
    XY = [list(map(int, input().split())) for _ in range(N)]
    XY.sort()

    INF = 10**18
    dp = [INF]*N
    dp[0] = 0
    que = [(0, 0)]
    while que:
        c, v = heapq.heappop(que)
        if dp[v] < c:
            continue
        for i in range(v+1, N):
            nc = c + sqrt((XY[i][0]-XY[v][0])**2 + (XY[i][1]-XY[v][1])**2) + 2**(i-v-1) - 1
            if dp[i] > nc:
                dp[i] = nc
                heapq.heappush(que, (nc, i))
    print(dp[-1])

if __name__ == "__main__":
    main()