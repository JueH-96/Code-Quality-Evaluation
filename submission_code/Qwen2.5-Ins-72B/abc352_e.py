import sys
from collections import defaultdict
from heapq import heappop, heappush

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

def kruskal(edges, n):
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    mst_weight = 0
    edges.sort(key=lambda x: x[2])
    for u, v, w in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_weight += w
    for i in range(1, n + 1):
        if find(parent, i) != find(parent, 1):
            return -1
    return mst_weight

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    index = 2
    edges = []
    
    for _ in range(m):
        k = int(data[index])
        c = int(data[index + 1])
        index += 2
        subset = list(map(int, data[index:index + k]))
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                edges.append((subset[i], subset[j], c))
        index += k
    
    result = kruskal(edges, n)
    print(result)

if __name__ == "__main__":
    main()