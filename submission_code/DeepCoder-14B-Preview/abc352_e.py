class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.rank = [1] * (size + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False  # already connected
        # Union by rank
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
        return True

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    edges = []
    
    for _ in range(M):
        K_i = int(input[ptr])
        ptr += 1
        C_i = int(input[ptr])
        ptr += 1
        S_i = list(map(int, input[ptr:ptr + K_i]))
        ptr += K_i
        
        a = S_i[0]
        for b in S_i[1:]:
            edges.append((C_i, a, b))
    
    # Sort edges by weight
    edges.sort()
    
    dsu = DSU(N)
    total = 0
    count = 0
    
    for C, u, v in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            total += C
            count += 1
            if count == N - 1:
                break
    
    if count == N - 1:
        print(total)
    else:
        print(-1)

if __name__ == '__main__':
    main()