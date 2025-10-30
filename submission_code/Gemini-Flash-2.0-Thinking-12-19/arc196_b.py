import sys

# Increase recursion depth for Union-Find
sys.setrecursionlimit(2000000)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.num_components = n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_components -= 1

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    MOD = 998244353

    row_a_counts = [0] * H
    col_a_counts = [0] * W
    b_cells = []

    for i in range(H):
        for j in range(W):
            if S[i][j] == 'A':
                row_a_counts[i] += 1
                col_a_counts[j] += 1
            else:
                b_cells.append((i, j))

    for count in row_a_counts:
        if count % 2 != 0:
            print(0)
            return
    for count in col_a_counts:
        if count % 2 != 0:
            print(0)
            return

    # If row and column counts of 'A' tiles are all even, solutions exist.
    # The number of valid placements corresponds to the number of solutions
    # for the edge variables (h_ij, v_ij) satisfying the constraints derived
    # from tile types and the no-dead-end condition.
    # These constraints form a system of linear equations over GF(2).
    # The number of solutions to this system is 2^k where k is the dimension of the solution space.
    # This dimension k equals the number of connected components in a specific graph.
    # The graph vertices are H for rows and W for columns. An edge connects row i and column j
    # if S[i][j] is 'B'. The number of solutions is 2^(number of connected components)
    # in this bipartite graph.

    # Vertices 0 to H-1 represent rows.
    # Vertices H to H+W-1 represent columns.
    uf = UnionFind(H + W)

    for r_idx, c_idx in b_cells:
        # Edge between row vertex r_idx and column vertex H + c_idx
        uf.union(r_idx, H + c_idx)

    print(pow(2, uf.num_components, MOD))

T = int(sys.stdin.readline())
for _ in range(T):
    solve()