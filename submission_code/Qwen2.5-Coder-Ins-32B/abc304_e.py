# YOUR CODE HERE
import sys
import threading
from collections import defaultdict, deque

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def is_connected(parent, x, y):
    return find(parent, x) == find(parent, y)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    M = int(input[idx + 1])
    idx += 2
    edges = []
    for _ in range(M):
        u = int(input[idx]) - 1
        v = int(input[idx + 1]) - 1
        edges.append((u, v))
        idx += 2
    K = int(input[idx])
    idx += 1
    forbidden = []
    for _ in range(K):
        x = int(input[idx]) - 1
        y = int(input[idx + 1]) - 1
        forbidden.append((x, y))
        idx += 2
    Q = int(input[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        p = int(input[idx]) - 1
        q = int(input[idx + 1]) - 1
        queries.append((p, q))
        idx += 2

    parent = list(range(N))
    rank = [0] * N

    for u, v in edges:
        union(parent, rank, u, v)

    forbidden_dict = defaultdict(set)
    for x, y in forbidden:
        forbidden_dict[x].add(y)
        forbidden_dict[y].add(x)

    results = []
    for p, q in queries:
        if is_connected(parent, p, q):
            results.append("No")
            continue
        temp_parent = parent[:]
        temp_rank = rank[:]
        union(temp_parent, temp_rank, p, q)
        connected = False
        for x, y in forbidden:
            if is_connected(temp_parent, x, y):
                connected = True
                break
        if connected:
            results.append("No")
        else:
            results.append("Yes")

    for result in results:
        print(result)

threading.Thread(target=main).start()