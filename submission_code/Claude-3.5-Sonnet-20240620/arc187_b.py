# YOUR CODE HERE
from collections import defaultdict

MOD = 998244353

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def count_components(N, edges):
    parent = list(range(N))
    rank = [0] * N
    for u, v in edges:
        union(parent, rank, u, v)
    return len(set(find(parent, i) for i in range(N)))

def solve(N, M, B):
    dp = [0] * (1 << N)
    dp[0] = 1

    for mask in range(1, 1 << N):
        edges = []
        unknown = []
        for i in range(N):
            if mask & (1 << i):
                for j in range(i + 1, N):
                    if mask & (1 << j):
                        if B[i] != -1 and B[j] != -1:
                            if B[i] <= B[j]:
                                edges.append((i, j))
                        elif B[i] == -1 or B[j] == -1:
                            unknown.append((i, j))

        components = count_components(N, edges)
        submask = mask
        while submask:
            if dp[submask] != 0:
                diff = bin(mask ^ submask).count('1')
                dp[mask] = (dp[mask] + dp[submask] * pow(M, diff, MOD) * components) % MOD
            submask = (submask - 1) & mask

    return dp[(1 << N) - 1]

N, M = map(int, input().split())
B = list(map(int, input().split()))

result = solve(N, M, B)
print(result)