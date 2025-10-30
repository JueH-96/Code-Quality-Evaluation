import sys
import itertools

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u, v, w))
    
    # Generate all possible spanning trees
    # Since N is small (up to 8), we can use brute force
    # We need to select N-1 edges that form a tree
    # To check if a set of edges forms a tree, we can use Union-Find
    
    # Union-Find data structure
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size + 1))
            self.size = [1] * (size + 1)
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            xroot = self.find(x)
            yroot = self.find(y)
            if xroot == yroot:
                return False
            if self.size[xroot] < self.size[yroot]:
                xroot, yroot = yroot, xroot
            self.parent[yroot] = xroot
            self.size[xroot] += self.size[yroot]
            return True
    
    min_cost = float('inf')
    # Iterate over all combinations of N-1 edges
    for selected_edges in itertools.combinations(range(M), N-1):
        uf = UnionFind(N)
        total = 0
        valid = True
        for idx in selected_edges:
            u, v, w = edges[idx]
            if not uf.union(u, v):
                valid = False
                break
            total += w
        if valid:
            current_cost = total % K
            if current_cost < min_cost:
                min_cost = current_cost
    print(min_cost)

if __name__ == "__main__":
    main()