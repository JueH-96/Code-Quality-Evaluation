def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        H = int(next(it))
        W = int(next(it))
    except StopIteration:
        return
    grid = [next(it).strip() for _ in range(H)]
    MOD = 998244353
    n = H * W

    # Build union-find arrays for green cells.
    # For each cell, if it is green ('#'), set it as its own parent, otherwise mark it with -1.
    parent = [-1] * n
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                parent[i * W + j] = i * W + j

    # Union-find functions (iterative find with path compression):
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        parent[ry] = rx

    # Union adjacent green cells.
    # We only need to look right and down to avoid double‐counting.
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#':
                continue
            idx = i * W + j
            if j + 1 < W and grid[i][j + 1] == '#':
                union(idx, i * W + (j + 1))
            if i + 1 < H and grid[i + 1][j] == '#':
                union(idx, (i + 1) * W + j)

    # Count the number of distinct connected components among the green cells.
    comp_set = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                comp_set.add(find(i * W + j))
    orig_components = len(comp_set)

    # Now, for each red cell ('.'), check its four neighbors and count how many distinct green
    # connected components are adjacent. When repainting a red cell green, its effect is that it
    # connects these adjacent green components. The new connectivity will be:
    #    new_components = (orig_components - (number of distinct adjacent green comp)) + 1.
    # Hence, the expected number is: orig_components + 1 - (average over red cells of (# distinct adjacent components)).
    red_count = 0
    total_adj_comp = 0
    # Check 4-neighborhood: up, right, down, left.
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':  # red cell
                red_count += 1
                adj_set = set()
                for di, dj in dirs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            adj_set.add(find(ni * W + nj))
                total_adj_comp += len(adj_set)
                
    # Expected new connectivity = orig_components + 1 - (total_adj_comp / red_count).
    # Writing it as a fraction:
    #    result = ((orig_components + 1) * red_count - total_adj_comp) / red_count.
    numerator = ((orig_components + 1) * red_count - total_adj_comp) % MOD
    # red_count is positive by the problem statement.
    inv_red = pow(red_count, MOD - 2, MOD)
    answer = (numerator * inv_red) % MOD

    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()