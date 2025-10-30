def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid_lines = input_data[2:]
    
    # We'll work in a 1D index space: index(i, j) = i*W + j
    # Union-Find/Disjoint Set implementation
    parent = list(range(H*W))
    rank_ = [0]*(H*W)
    
    def find(x):
        # Iterative path compression
        st = []
        while parent[x] != x:
            st.append(x)
            x = parent[x]
        # Path-compress
        for y in st:
            parent[y] = x
        return x
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank_[ra] < rank_[rb]:
            parent[ra] = rb
        elif rank_[ra] > rank_[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank_[ra] += 1

    # Read the grid
    # '.' -> red, '#' -> green
    grid = grid_lines

    # Directions for adjacency (right, down) - to avoid double counting
    # We'll union only green neighbors
    for i in range(H):
        row = grid[i]
        for j in range(W):
            if row[j] == '#':
                idx = i*W + j
                # Check right
                if j+1 < W and row[j+1] == '#':
                    union(idx, i*W + (j+1))
                # Check down
                if i+1 < H and grid[i+1][j] == '#':
                    union(idx, (i+1)*W + j)

    # Count how many distinct green components
    # We'll iterate over all green cells, find their root, gather in a set
    green_roots = set()
    red_cells = []
    for i in range(H):
        row = grid[i]
        for j in range(W):
            if row[j] == '#':
                green_roots.add(find(i*W + j))
            else:
                # It's a red cell
                red_cells.append((i, j))
    base_components = len(green_roots)
    # Number of red cells
    R = len(red_cells)
    
    # For each red cell, compute how many distinct neighbor components it touches
    # The new number of green components after painting = base_components + 1 - k
    # where k = number of distinct green-connected components among neighbors
    adj = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    M = 998244353
    
    sum_new = 0
    for (r, c) in red_cells:
        neighbor_roots = set()
        for dr, dc in adj:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] == '#':
                    neighbor_roots.add(find(nr*W + nc))
        k = len(neighbor_roots)
        # new_component_count = base_components + 1 - k
        sum_new += (base_components + 1 - k)
    
    sum_new_mod = sum_new % M
    
    # We want (sum_new / R) mod M.
    # Because M is prime, we can use Fermat's little theorem for modular inverse of R mod M.
    # inv(R) = R^(M-2) mod M
    # Then result = sum_new * inv(R) mod M
    # We'll compute that and print.
    
    # Quick exponentiation
    def modpow(base, exp, mod):
        result = 1
        cur = base % mod
        e = exp
        while e > 0:
            if e & 1:
                result = (result * cur) % mod
            cur = (cur * cur) % mod
            e >>= 1
        return result
    
    inv_R = modpow(R, M-2, M)  # R^(M-2) mod M
    ans = (sum_new_mod * inv_R) % M
    print(ans)