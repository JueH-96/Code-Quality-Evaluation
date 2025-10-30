import sys
from collections import defaultdict

def find_parent(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find_parent(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find_parent(parent, x)
    yroot = find_parent(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def are_connected(parent, x, y):
    return find_parent(parent, x) == find_parent(parent, y)

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    edges = []
    for _ in range(M):
        u = int(data[index])
        v = int(data[index + 1])
        edges.append((u, v))
        index += 2

    K = int(data[index])
    index += 1

    pairs = []
    for _ in range(K):
        x = int(data[index])
        y = int(data[index + 1])
        pairs.append((x, y))
        index += 2

    Q = int(data[index])
    index += 1

    queries = []
    for _ in range(Q):
        p = int(data[index])
        q = int(data[index + 1])
        queries.append((p, q))
        index += 2

    parent = list(range(N + 1))
    rank = [0] * (N + 1)

    for u, v in edges:
        union(parent, rank, u, v)

    results = []
    for p, q in queries:
        if are_connected(parent, p, q):
            results.append("No")
        else:
            results.append("Yes")

    for result in results:
        print(result)

if __name__ == "__main__":
    main()