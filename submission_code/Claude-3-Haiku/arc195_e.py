import sys

MOD = 998244353

def dfs(parent, node, weights, distances):
    total_distance = 0
    for child in range(2, node + 1):
        if parent[child] == node:
            total_distance += weights[child] + dfs(parent, child, weights, distances)
    distances[node] = total_distance
    return total_distance

def solve():
    N, Q = map(int, input().split())
    weights = [0] + list(map(int, input().split()))
    for _ in range(Q):
        u, v = map(int, input().split())
        parent = [0] * (N + 1)
        distances = [0] * (N + 1)
        dfs(parent, 1, weights, distances)
        total_distance = 0
        for p in range(2, N + 1):
            parent[p] = p - 1
            total_distance += (N - 1) * distances[p]
            for i in range(2, p):
                if parent[i] == p - 1:
                    total_distance -= distances[i]
        print(total_distance % MOD)

solve()