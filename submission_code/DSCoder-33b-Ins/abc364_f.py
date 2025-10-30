# YOUR CODE HERE
from heapq import heappop, heappush

def solve():
    N, Q = map(int, input().split())
    edges = []
    for _ in range(Q):
        L, R, C = map(int, input().split())
        edges.append((L, R, C))

    edges.sort(key=lambda x: x[2])

    parent = list(range(N+Q+1))
    rank = [0] * (N+Q+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        xr = find(x)
        yr = find(y)
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr] += 1

    mst = []
    for L, R, C in edges:
        for i in range(L, R+1):
            if find(i) != find(N+Q+1):
                union(i, N+Q+1)
                mst.append(C)
                break

    if len(mst) != N+Q-1:
        print(-1)
    else:
        print(sum(mst))

solve()