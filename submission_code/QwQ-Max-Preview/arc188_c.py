import sys
from collections import defaultdict

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))  # 1-based indexing
        self.rank = [0] * (size + 1)
        self.parity = [0] * (size + 1)  # parity from node to parent

    def find(self, u):
        if self.parent[u] != u:
            orig_parent = self.parent[u]
            self.parent[u], p = self.find(orig_parent)
            self.parity[u] ^= p
        return self.parent[u], self.parity[u]

    def union(self, u, v, expected_parity):
        root_u, parity_u = self.find(u)
        root_v, parity_v = self.find(v)
        if root_u == root_v:
            return
        if self.rank[root_u] < self.rank[root_v]:
            root_u, root_v = root_v, root_u
            parity_u, parity_v = parity_v, parity_u
            u, v = v, u
            expected_parity ^= 1
        self.parent[root_v] = root_u
        self.parity[root_v] = parity_u ^ parity_v ^ expected_parity
        if self.rank[root_u] == self.rank[root_v]:
            self.rank[root_u] += 1

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    testimonies = defaultdict(list)
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        testimonies[A].append( (B, C) )
    
    dsu = DSU(N)
    confused = ['0'] * N  # 0-based index for villagers 0..N-1

    for A in range(1, N+1):
        t_list = testimonies.get(A, [])
        if not t_list:
            continue

        valid0 = True
        for B, C in t_list:
            expected_parity = 0 ^ C
            rootA, parityA = dsu.find(A)
            rootB, parityB = dsu.find(B)
            if rootA == rootB:
                current_parity = parityA ^ parityB
                if current_parity != expected_parity:
                    valid0 = False
                    break

        valid1 = True
        for B, C in t_list:
            expected_parity = 1 ^ C
            rootA, parityA = dsu.find(A)
            rootB, parityB = dsu.find(B)
            if rootA == rootB:
                current_parity = parityA ^ parityB
                if current_parity != expected_parity:
                    valid1 = False
                    break

        if valid0:
            for B, C in t_list:
                expected_parity = 0 ^ C
                dsu.union(A, B, expected_parity)
            confused[A-1] = '0'
        elif valid1:
            for B, C in t_list:
                expected_parity = 1 ^ C
                dsu.union(A, B, expected_parity)
            confused[A-1] = '1'
        else:
            print(-1)
            return

    print(''.join(confused))

if __name__ == '__main__':
    main()