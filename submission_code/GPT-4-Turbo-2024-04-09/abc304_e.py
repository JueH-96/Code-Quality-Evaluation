import sys
input = sys.stdin.read
from sys import setrecursionlimit
from collections import defaultdict, deque

setrecursionlimit(10**6)

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    graph = defaultdict(list)
    
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        graph[u].append(v)
        graph[v].append(u)
    
    K = int(data[index])
    index += 1
    
    forbidden_pairs = []
    for _ in range(K):
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        forbidden_pairs.append((x, y))
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        p = int(data[index])
        index += 1
        q = int(data[index])
        index += 1
        queries.append((p, q))
    
    # Use Union-Find (Disjoint Set Union) to manage connected components
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
    
    # Initially union all connected nodes in the graph
    for u in graph:
        for v in graph[u]:
            union(u, v)
    
    # Check forbidden pairs
    forbidden_dict = {}
    for x, y in forbidden_pairs:
        rootX = find(x)
        rootY = find(y)
        if rootX == rootY:
            # This should not happen as per problem statement
            continue
        if rootX > rootY:
            rootX, rootY = rootY, rootX
        forbidden_dict[(rootX, rootY)] = True
    
    results = []
    for p, q in queries:
        rootP = find(p)
        rootQ = find(q)
        if rootP == rootQ:
            results.append("No")
        else:
            if rootP > rootQ:
                rootP, rootQ = rootQ, rootP
            if (rootP, rootQ) in forbidden_dict:
                results.append("No")
            else:
                results.append("Yes")
    
    print("
".join(results))

if __name__ == "__main__":
    main()