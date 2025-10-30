import sys
import heapq

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
    
    INF = float('inf')
    full_mask = (1 << N) - 1

    dp = [[INF] * (1 << N) for _ in range(N + 1)]
    for u in range(1, N + 1):
        dp[u][1 << (u - 1)] = 0
    
    # Generate all masks and sort them by the number of bits
    masks = []
    for mask in range(1 << N):
        cnt = bin(mask).count('1')
        masks.append((cnt, mask))
    masks.sort()
    
    for cnt, mask in masks:
        for u in range(1, N + 1):
            if dp[u][mask] == INF:
                continue
            for v, w in adj[u]:
                new_mask = mask | (1 << (v - 1))
                new_cost = dp[u][mask] + w
                if new_cost < dp[v][new_mask]:
                    dp[v][new_mask] = new_cost
    
    min_total = INF
    for u in range(1, N + 1):
        if dp[u][full_mask] < min_total:
            min_total = dp[u][full_mask]
    
    if min_total == INF:
        print("No")
    else:
        print(min_total)

if __name__ == "__main__":
    main()