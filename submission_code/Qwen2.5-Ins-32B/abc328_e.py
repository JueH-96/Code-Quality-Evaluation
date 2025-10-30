# YOUR CODE HERE
import itertools

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

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

def kruskalMST(N, edges, K):
    result = []
    i = 0
    e = 0
    edges = sorted(edges, key=lambda item: item[2])
    parent = []
    rank = []
    for node in range(N):
        parent.append(node)
        rank.append(0)
    while e < N - 1:
        u, v, w = edges[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(parent, rank, x, y)
    return sum([w for u, v, w in result]) % K

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    edges = []
    for _ in range(M):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        w = int(data[index])
        index += 1
        edges.append((u, v, w))
    print(kruskalMST(N, edges, K))

solve()