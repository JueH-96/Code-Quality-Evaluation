import sys

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]
        return True

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    edges = []
    for i in range(1, Q+1):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        v = N + i
        edges.append( (C, v, L) )
        edges.append( (C, v, R + 1) )

    edges.sort()

    max_node = N + Q
    dsu = DSU(max_node + 2)  # To handle up to N+Q+1 if needed
    total = 0

    for c, u, v in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            total += c

    root = dsu.find(1)
    # Check original nodes 1..N
    for node in range(2, N+1):
        if dsu.find(node) != root:
            print(-1)
            return
    # Check new nodes N+1 ... N+Q
    for node in range(N+1, N + Q + 1):
        if dsu.find(node) != root:
            print(-1)
            return

    print(total)

if __name__ == "__main__":
    main()