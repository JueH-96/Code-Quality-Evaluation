import sys
from itertools import combinations

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.rank = [0] * (size + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    
    edges = []
    for _ in range(M):
        u = int(input[idx]); idx +=1
        v = int(input[idx]); idx +=1
        w = int(input[idx]); idx +=1
        edges.append((u, v, w))
    
    min_mod = K  # Initialize with a value larger than possible minimum
    
    for comb in combinations(edges, N-1):
        dsu = DSU(N)
        valid = True
        for u, v, w in comb:
            if dsu.find(u) == dsu.find(v):
                valid = False
                break
            dsu.union(u, v)
        if not valid:
            continue
        
        root = dsu.find(1)
        connected = True
        for node in range(2, N+1):
            if dsu.find(node) != root:
                connected = False
                break
        if connected:
            total = sum(w for u, v, w in comb)
            mod_total = total % K
            if mod_total < min_mod:
                min_mod = mod_total
    
    print(min_mod)

if __name__ == "__main__":
    main()