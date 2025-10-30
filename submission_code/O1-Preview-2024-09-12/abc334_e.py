# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]

    parent = {}
    size = {}

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if size[u_root] < size[v_root]:
            parent[u_root] = v_root
            size[v_root] += size[u_root]
        else:
            parent[v_root] = u_root
            size[u_root] += size[v_root]

    # Initialize Union-Find
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                parent[(i,j)] = (i,j)
                size[(i,j)] = 1

    # Union adjacent green cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                for dir in [(-1,0),(0,-1)]:
                    ni = i + dir[0]
                    nj = j + dir[1]
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        union((i,j), (ni,nj))

    # Count initial number of green connected components
    roots = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                roots.add(find((i,j)))
    N0 = len(roots)

    total_number_of_red_cells = 0
    N1_sum = 0

    mod = 998244353

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_number_of_red_cells += 1
                S = set()
                for dir in [(-1,0),(0,-1),(1,0),(0,1)]:
                    ni = i + dir[0]
                    nj = j + dir[1]
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        S.add(find((ni, nj)))
                k = len(S)
                if k == 0:
                    N1 = N0 + 1
                else:
                    N1 = N0 - (k - 1)
                N1_sum += N1

    # Compute expected value E = N1_sum / total_number_of_red_cells
    P = N1_sum
    Q = total_number_of_red_cells

    # Simplify fraction P/Q by dividing by GCD
    from math import gcd
    g = gcd(P, Q)
    P //= g
    Q //= g

    # Compute modular inverse of Q modulo 998244353
    R = P * pow(Q, mod - 2, mod)
    R %= mod

    print(R)

# For faster I/O and threading for large inputs
threading.Thread(target=main).start()