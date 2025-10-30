import sys
import heapq

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True
        return False

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    first_line = data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    
    edges = []
    
    index = 1
    for _ in range(M):
        k_c = data[index].split()
        K_i = int(k_c[0])
        C_i = int(k_c[1])
        vertices = list(map(int, data[index + 1].split()))
        
        for j in range(K_i):
            for l in range(j + 1, K_i):
                u = vertices[j] - 1
                v = vertices[l] - 1
                edges.append((C_i, u, v))
        
        index += 2
    
    # Kruskal's algorithm to find the minimum spanning tree
    edges.sort()  # Sort edges by weight
    uf = UnionFind(N)
    
    total_weight = 0
    edge_count = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            total_weight += weight
            edge_count += 1
            if edge_count == N - 1:  # We need exactly N-1 edges for a spanning tree
                break
    
    if edge_count == N - 1:
        print(total_weight)
    else:
        print(-1)

if __name__ == "__main__":
    main()