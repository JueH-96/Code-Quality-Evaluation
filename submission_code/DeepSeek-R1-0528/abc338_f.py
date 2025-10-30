import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    INF = 10**18
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    idx = 2
    for i in range(m):
        u = int(data[idx])
        v = int(data[idx+1])
        w = int(data[idx+2])
        idx += 3
        u -= 1
        v -= 1
        if w < dist[u][v]:
            dist[u][v] = w

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    dp = [[INF] * n for _ in range(1<<n)]
    
    for i in range(n):
        dp[1<<i][i] = 0
        
    total_mask = (1<<n)
    for mask in range(total_mask):
        for i in range(n):
            if dp[mask][i] == INF:
                continue
            if mask & (1 << i) == 0:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                if dist[i][j] == INF:
                    continue
                new_mask = mask | (1<<j)
                new_val = dp[mask][i] + dist[i][j]
                if new_val < dp[new_mask][j]:
                    dp[new_mask][j] = new_val
                    
    ans = min(dp[total_mask-1])
    if ans >= INF:
        print("No")
    else:
        print(ans)

if __name__ == "__main__":
    main()