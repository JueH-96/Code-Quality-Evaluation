import sys
from collections import defaultdict

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

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N, M = int(data[index]), int(data[index+1])
    index += 2
    
    edges = []
    for _ in range(M):
        u, v = int(data[index]), int(data[index+1])
        edges.append((u, v))
        index += 2
    
    K = int(data[index])
    index += 1
    forbidden_pairs = []
    for _ in range(K):
        x, y = int(data[index]), int(data[index+1])
        forbidden_pairs.append((x, y))
        index += 2
    
    Q = int(data[index])
    index += 1
    queries = []
    for _ in range(Q):
        p, q = int(data[index]), int(data[index+1])
        queries.append((p, q))
        index += 2
    
    parent = [i for i in range(N+1)]
    rank = [0] * (N+1)
    
    for u, v in edges:
        union(parent, rank, u, v)
    
    forbidden_sets = defaultdict(set)
    for x, y in forbidden_pairs:
        forbidden_sets[find(parent, x)].add(find(parent, y))
    
    for p, q in queries:
        if find(parent, p) == find(parent, q) or find(parent, q) in forbidden_sets[find(parent, p)]:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()