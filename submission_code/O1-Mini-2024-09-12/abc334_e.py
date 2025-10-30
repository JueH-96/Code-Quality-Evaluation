# YOUR CODE HERE
import sys
import sys
sys.setrecursionlimit(1 << 25)
def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353

    H, W, *rest = input().split()
    H = int(H)
    W = int(W)
    grid = rest[:H]

    parent = [i for i in range(H * W)]
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        parent[v_root] = u_root

    # Directions: up, left
    dirs = [(-1,0),(0,-1),(1,0),(0,1)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                current = i * W + j
                for di, dj in [(-1,0),(0,-1),(1,0),(0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        neighbor = ni * W + nj
                        union(current, neighbor)

    # Find unique parents for green cells
    parents = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                parents.add(find(i * W + j))
    C_init = len(parents)

    # Collect red cells
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i,j))
    R = len(red_cells)

    sum_k = 0
    for i,j in red_cells:
        adj_parents = []
        for di, dj in [(-1,0),(0,-1),(1,0),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                adj_parents.append(find(ni * W + nj))
        # Count unique parents
        unique = set(adj_parents)
        k = len(unique)
        sum_k += k

    P = (C_init +1) * R - sum_k
    P %= MOD
    Q = R
    Q_inv = pow(Q, MOD -2, MOD)
    E = (P * Q_inv) % MOD
    print(E)

if __name__ == "__main__":
    main()