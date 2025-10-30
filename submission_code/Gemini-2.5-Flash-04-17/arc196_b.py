import sys

def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

MOD = 998244353

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.diff = [0] * n # diff[i] stores value[i] XOR value[parent[i]]
        self.components = n
        self.is_consistent = True

    def find(self, i):
        if self.parent[i] == i:
            return (i, 0)
        root, root_diff = self.find(self.parent[i])
        self.parent[i] = root
        self.diff[i] ^= root_diff
        return (root, self.diff[i])

    def union(self, i, j, c): # implies value[i] XOR value[j] = c
        if not self.is_consistent:
            return

        root_i, diff_i = self.find(i)
        root_j, diff_j = self.find(j)

        if root_i != root_j:
            self.parent[root_i] = root_j
            self.diff[root_i] = diff_i ^ diff_j ^ c
            self.components -= 1
        else:
            # Check consistency
            if (diff_i ^ diff_j) != c:
                self.is_consistent = False

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    row_a_counts = [0] * H
    col_a_counts = [0] * W
    delta = [[0] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):
            if S[i][j] == 'A':
                row_a_counts[i] += 1
                col_a_counts[j] += 1
                delta[i][j] = 1

    for count in row_a_counts:
        if count % 2 != 0:
            print(0)
            return
    for count in col_a_counts:
        if count % 2 != 0:
            print(0)
            return

    # Calculate VSum_val[i][j] = v_i_j XOR v_0_j
    # v_i_j = v_{i-1,j} ^ \delta_{ij} for i > 0
    # v_i_j ^ v_0_j = v_{i-1,j} ^ v_0_j ^ \delta_{ij}
    # VSum_val[i][j] = VSum_val[i-1][j] ^ \delta_{ij} for i > 0. VSum_val[0][j]=0.
    # This gives VSum_val[i][j] = \bigoplus_{k=1}^i \delta_{kj} for i >= 1, VSum_val[0][j]=0.
    VSum_val = [[0] * W for _ in range(H)]
    for j in range(W):
        for i in range(1, H):
            VSum_val[i][j] = VSum_val[i-1][j] ^ delta[i][j]

    # Calculate HSum_val[i][j] = h_i_j XOR h_i_0
    # h_i_j = h_{i,j-1} ^ \delta_{ij} for j > 0
    # h_i_j ^ h_i_0 = h_{i,j-1} ^ h_i_0 ^ \delta_{ij}
    # HSum_val[i][j] = HSum_val[i][j-1] ^ \delta_{ij} for j > 0. HSum_val[i][0]=0.
    # This gives HSum_val[i][j] = \bigoplus_{k=1}^j \delta_{ik} for j >= 1, HSum_val[i][0]=0.
    HSum_val = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(1, W):
            HSum_val[i][j] = HSum_val[i][j-1] ^ delta[i][j]

    # DSU on W+H variables (v_0_j for j=0..W-1, h_i_0 for i=0..H-1)
    # v_0_j is node j, h_i_0 is node W+i
    dsu = DSU(W + H)

    for i in range(H):
        for j in range(W):
            if S[i][j] == 'B':
                # Constraint for S_ij == 'B': v_{(i-1)%H, j} ^ h_{i, (j-1)%W} = 1
                # Express in terms of base variables v_0_j and h_i_0:
                # v_{(i-1)%H, j} = v_0_j ^ (v_{(i-1)%H, j} ^ v_0_j)
                # The term v_{k,j} ^ v_{0,j} is VSum_val[k][j] for k >= 1, and 0 for k=0.
                # Correct term is $\bigoplus_{p=1}^k \delta_{pj}$ for $k \ge 1$, $0$ for $k=0$.
                # Let $idx\_im1 = (i-1+H)\%H$.
                # The term $v_{idx\_im1, j} \oplus v_{0,j}$ should be calculated correctly.
                # If idx_im1 == 0, term is 0.
                # If idx_im1 > 0, term is VSum_val[idx_im1][j].
                # This is exactly VSum_val[(i-1+H)%H][j] from our calculation logic.

                v_idx = j # corresponds to v_0_j
                h_idx = W + i # corresponds to h_i_0
                
                idx_im1 = (i - 1 + H) % H
                idx_jm1 = (j - 1 + W) % W
                
                vs_term_diff = VSum_val[idx_im1][j]
                hs_term_diff = HSum_val[i][idx_jm1]

                # Equation: (v_0_j ^ vs_term_diff) ^ (h_i_0 ^ hs_term_diff) = 1
                # v_0_j ^ h_i_0 = 1 ^ vs_term_diff ^ hs_term_diff
                c = 1 ^ vs_term_diff ^ hs_term_diff
                
                dsu.union(v_idx, h_idx, c)
                
                if not dsu.is_consistent:
                    print(0)
                    return

    print(power(2, dsu.components, MOD))

T = int(sys.stdin.readline())
for _ in range(T):
    solve()