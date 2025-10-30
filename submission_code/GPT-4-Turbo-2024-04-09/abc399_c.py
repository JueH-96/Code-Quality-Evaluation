import sys
input = sys.stdin.read
from sys import setrecursionlimit
from collections import defaultdict

setrecursionlimit(10**6)

def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x], parent)
    return parent[x]

def union(x, y, parent, rank):
    rootX = find(x, parent)
    rootY = find(y, parent)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1
        return True
    return False

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        edges.append((u, v))
    
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    # Count the number of edges that are part of any cycle
    redundant_edges = 0
    for u, v in edges:
        if not union(u, v, parent, rank):
            redundant_edges += 1
    
    print(redundant_edges)

if __name__ == "__main__":
    main()