import sys
import itertools

def floyd_warshall(graph, n):
    dist = [[float('inf')] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0
    for u in range(n):
        for v, w in graph[u]:
            dist[u][v] = min(dist[u][v], w)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def tsp(graph, n):
    dist = floyd_warshall(graph, n)
    
    dp = [[float('inf')] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = 0
    
    for mask in range(1 << n):
        for u in range(n):
            if (mask & (1 << u)) == 0:
                continue
            for v in range(n):
                if (mask & (1 << v)) != 0:
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
    
    min_cost = float('inf')
    for u in range(n):
        min_cost = min(min_cost, dp[(1 << n) - 1][u])
    
    return min_cost

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    n, m = map(int, data[0].split())
    graph = [[] for _ in range(n)]
    
    for i in range(1, m + 1):
        u, v, w = map(int, data[i].split())
        graph[u - 1].append((v - 1, w))
    
    min_weight = tsp(graph, n)
    
    if min_weight == float('inf'):
        print("No")
    else:
        print(min_weight)

if __name__ == "__main__":
    main()