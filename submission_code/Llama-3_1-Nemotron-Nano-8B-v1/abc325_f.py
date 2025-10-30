import sys
import math

def main():
    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L1, C1, K1 = map(int, sys.stdin.readline().split())
    L2, C2, K2 = map(int, sys.stdin.readline().split())

    sections = []
    for d in D:
        options = []
        x_max = min((d + L1 - 1) // L1, K1)
        for x in range(x_max + 1):
            remaining = d - x * L1
            if remaining <= 0:
                y = 0
            else:
                y = (remaining + L2 - 1) // L2
            if y > K2:
                continue
            cost = x * C1 + y * C2
            options.append((x, y, cost))
        options.sort()
        pruned = []
        min_cost = float('inf')
        for x, y, cost in options:
            if cost < min_cost:
                pruned.append((x, y, cost))
                min_cost = cost
        sections.append(pruned)
    
    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0
    for section in sections:
        for x, y, cost in section:
            for i in range(K1, -1, -1):
                for j in range(K2, -1, -1):
                    if dp[i][j] != INF:
                        ni, nj = i + x, j + y
                        if ni <= K1 and nj <= K2:
                            if dp[ni][nj] > dp[i][j] + cost:
                                dp[ni][nj] = dp[i][j] + cost
    min_cost = INF
    for i in range(K1 + 1):
        for j in range(K2 + 1):
            if dp[i][j] < min_cost:
                min_cost = dp[i][j]
    print(-1 if min_cost == INF else min_cost)

if __name__ == '__main__':
    main()