import sys

# Increase recursion depth for UnionFind
sys.setrecursionlimit(2000000)

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

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.parity = [0] * n # parity[i] stores parity of (variable_i XOR variable_root(i))
        self.components = n
        self.consistent = True

    def find(self, i):
        if self.parent[i] == i:
            return i, 0
        root, root_parity = self.find(self.parent[i])
        self.parent[i] = root
        self.parity[i] = (self.parity[i] + root_parity) % 2 # XOR sum
        return root, self.parity[i]

    # Unite sets containing u and v, with constraint variable_u XOR variable_v = k
    def unite(self, u, v, k):
        root_u, parity_u = self.find(u)
        root_v, parity_v = self.find(v)
        if root_u != root_v:
            self.parent[root_v] = root_u
            # We want variable_root_v XOR variable_root_u = d
            # variable_u = variable_root_u XOR parity_u
            # variable_v = variable_root_v XOR parity_v = (variable_root_u XOR d) XOR parity_v
            # variable_u XOR variable_v = parity_u XOR (variable_root_u XOR d XOR parity_v)
            #                             = variable_root_u XOR parity_u XOR variable_root_u XOR d XOR parity_v
            # variable_u XOR variable_v = d XOR parity_u XOR parity_v
            # We are given variable_u XOR variable_v = k
            # So k = d XOR parity_u XOR parity_v
            # d = k XOR parity_u XOR parity_v
            self.parity[root_v] = (k + parity_u + parity_v) % 2 # XOR sum
            self.components -= 1
        else:
            # Check consistency: variable_u XOR variable_v = (variable_root_u XOR parity_u) XOR (variable_root_v XOR parity_v)
            # root_u == root_v, so variable_u XOR variable_v = parity_u XOR parity_v
            if (parity_u + parity_v) % 2 != k: # XOR sum
                self.consistent = False

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    a = [[0] * W for _ in range(H)]
    row_A_counts = [0] * H
    col_A_counts = [0] * W
    B_cells = []

    for i in range(H):
        for j in range(W):
            if S[i][j] == 'A':
                a[i][j] = 1
                row_A_counts[i] += 1
                col_A_counts[j] += 1
            else:
                a[i][j] = 0
                B_cells.append((i, j))

    for count in row_A_counts:
        if count % 2 != 0:
            print(0)
            return
    for count in col_A_counts:
        if count % 2 != 0:
            print(0)
            return

    # Calculate prefix XOR sums for A counts
    # LA_i[j] = XOR sum a[i][k] for k from 0 to j-1
    # TA_j[i] = XOR sum a[k][j] for k from 0 to i-1
    LA = [[0] * (W + 1) for _ in range(H)]
    TA = [[0] * (H + 1) for _ in range(W)]

    for i in range(H):
        for j in range(W):
            LA[i][j+1] = (LA[i][j] + a[i][j]) % 2

    for j in range(W):
        for i in range(H):
            TA[j][i+1] = (TA[j][i] + a[i][j]) % 2

    # System x_i XOR y_j = K_ij (mod 2) for B cells (i, j)
    # Variables x_0..x_{H-1}, y_0..y_{W-1}
    # Using nodes 0..H-1 for x_i, H..H+W-1 for y_j
    # Equation for cell (i,j) with 'B': v_ij != h_ij
    # v_ij = x_i XOR LA_i[j]
    # h_ij = y_j XOR TA_j[i]
    # x_i XOR LA_i[j] != y_j XOR TA_j[i]
    # x_i XOR y_j != LA_i[j] XOR TA_j[i]
    # x_i XOR y_j = 1 XOR (LA_i[j] XOR TA_j[i])
    # K_ij = 1 XOR LA_i[j] XOR TA_j[i]

    uf = UnionFind(H + W)

    for i, j in B_cells:
        # K_ij = 1 XOR LA_i[j] XOR TA_j[i]
        # Note: LA[i][j] represents the XOR sum of a[i][0]...a[i][j-1]
        # Note: TA[j][i] represents the XOR sum of a[0][j]...a[i-1][j]
        K_ij = (1 + LA[i][j] + TA[j][i]) % 2
        
        u_node = i       # Node for x_i
        v_node = H + j   # Node for y_j

        uf.unite(u_node, v_node, K_ij)
        
        if not uf.consistent:
            print(0)
            return
    
    # Number of solutions is 2 ^ (number of connected components)
    print(power(2, uf.components, MOD))

T = int(sys.stdin.readline())
for _ in range(T):
    solve()