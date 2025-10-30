import sys
import itertools

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u-1, v-1, w))  # converting to 0-based index
    
    # Generate all possible spanning trees
    # Since N is small (up to 8), we can use a brute-force approach
    # We need to select N-1 edges that form a tree
    # To find all possible combinations of N-1 edges that form a tree, we can use the following approach:
    # 1. Generate all combinations of N-1 edges
    # 2. Check if the selected edges form a connected graph with N-1 edges (which is a tree)
    
    # To check if a set of edges forms a tree, we can use Union-Find data structure
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.size = [1] * size
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]  # Path compression
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
    for comb in itertools.combinations(range(M), N-1):
        uf = UnionFind(N)
        total = 0
        valid = True
        for idx in comb:
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