import sys
from heapq import heappop, heappush

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

def kruskalMST(N, edges):
    result = []
    i = 0
    e = 0
    edges.sort(key=lambda item: item[2])
    parent = []
    rank = []
    for node in range(N+1):
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
    return result

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    for _ in range(M):
        K = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        vertices = list(map(int, data[index:index+K]))
        index += K
        for i in range(K):
            for j in range(i+1, K):
                edges.append((vertices[i], vertices[j], C))
    
    mst = kruskalMST(N, edges)
    if len(mst) < N - 1:
        print(-1)
    else:
        print(sum(w for u, v, w in mst))

if __name__ == "__main__":
    main()