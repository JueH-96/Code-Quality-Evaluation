import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # 1-based indexing
        self.rank = [1] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
            self.rank[y_root] += self.rank[x_root]
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += self.rank[y_root]
        return True

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    ops = []
    for i in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        ops.append((C, L, R, i))
    # Sort operations by cost
    ops.sort()
    dsu = DSU(N + Q)
    total = 0
    for C, L, R, i in ops:
        added_node = N + i + 1  # since i starts from 0, but original nodes are 1..N, added nodes are N+1..N+Q
        # Find roots of L and R in original nodes
        root_L = dsu.find(L)
        root_R = dsu.find(R)
        # Check if connected to added_node
        connected_L = dsu.find(added_node) == root_L
        connected_R = dsu.find(added_node) == root_R
        if not connected_L:
            dsu.union(added_node, root_L)
            total += C
        if not connected_R:
            dsu.union(added_node, root_R)
            total += C
    # Check if all original nodes are connected
    if N == 0:
        print(0)
        return
    root = dsu.find(1)
    for j in range(2, N+1):
        if dsu.find(j) != root:
            print(-1)
            return
    print(total)

if __name__ == "__main__":
    main()