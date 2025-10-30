import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # Using 1-based indexing
        self.rank = [1] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        # Union by rank
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

def main():
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx +=1
    m = int(input[idx])
    idx +=1
    dsu = DSU(n)
    for _ in range(m):
        a = int(input[idx])
        idx +=1
        b = int(input[idx])
        idx +=1
        dsu.union(a, b)
    
    component_size = {}
    for i in range(1, n+1):
        root = dsu.find(i)
        if root in component_size:
            component_size[root] +=1
        else:
            component_size[root] =1
    
    total =0
    for size in component_size.values():
        total += size * (size-1) //2
    print(total - m)

if __name__ == "__main__":
    main()