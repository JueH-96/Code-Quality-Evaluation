import sys

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
        self.parity = [0] * size  # XOR to parent.

    def find(self, u):
        if self.parent[u] != u:
            orig_parent = self.parent[u]
            self.find(self.parent[u])  # path compression
            self.parity[u] ^= self.parity[orig_parent]
            self.parent[u] = self.parent[orig_parent]
        return self.parent[u], self.parity[u]

    def union(self, u, v, x):
        ru, pu = self.find(u)
        rv, pv = self.find(v)
        if ru == rv:
            return (pu ^ pv) == x
        # Merge smaller tree into larger
        if self.rank[ru] < self.rank[rv]:
            ru, rv = rv, ru
            pu, pv = pv, pu
            u, v = v, u
        self.parent[rv] = ru
        self.parity[rv] = (pu ^ pv) ^ x
        if self.rank[ru] == self.rank[rv]:
            self.rank[ru] += 1
        return True

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    dsu = DSU(2 * N)
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        x_B = B - 1
        d_A = N + (A - 1)
        if not dsu.union(x_B, d_A, C):
            print(-1)
            return

    res = []
    for i in range(N):
        _, px = dsu.find(i)
        _, pd = dsu.find(N + i)
        res.append(str(px ^ pd))
    print(''.join(res))

if __name__ == '__main__':
    main()