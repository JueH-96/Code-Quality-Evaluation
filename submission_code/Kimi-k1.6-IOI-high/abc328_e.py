import itertools

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))  # 1-based indexing
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    edges = []
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        w = int(input[ptr])
        ptr += 1
        edges.append((u, v, w))
    
    min_cost = K  # Initialize to a value larger than maximum possible mod (K-1)
    
    # Iterate through all combinations of N-1 edges
    for combo in itertools.combinations(edges, N-1):
        dsu = DSU(N)
        for u, v, w in combo:
            dsu.union(u, v)
        # Check if all nodes are connected
        root = dsu.find(1)
        valid = True
        for i in range(2, N+1):
            if dsu.find(i) != root:
                valid = False
                break
        if valid:
            total = sum(w for u, v, w in combo)
            current_mod = total % K
            if current_mod < min_cost:
                min_cost = current_mod
    
    print(min_cost)

if __name__ == "__main__":
    main()