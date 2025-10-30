def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid_chars = input_data[2:]

    # We will treat '.' as red and '#' as green
    # We'll assign each cell a unique index idx = row*W + col

    # DSU (Disjoint Set Union) / Union-Find implementation
    parent = []
    rank = []

    def dsu_init(n):
        # Initialize DSU with n elements
        for _ in range(n):
            parent.append(-1)
            rank.append(0)

    def find(x):
        # Path-compression find
        if parent[x] < 0:
            return x
        parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        # Union by rank
        a = find(a)
        b = find(b)
        if a == b:
            return False
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        return True

    # Prepare DSU for H*W cells
    n = H * W
    dsu_init(n)

    # Convert grid characters into a 2D list for convenience
    grid = [list(row) for row in grid_chars]

    # Directions for adjacency (up, down, left, right)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # We'll union all adjacent green cells
    # Also keep track of how many green cells there are
    green_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':  # green cell
                green_count += 1
                idx = i * W + j
                # union with neighbors if they are green
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        union(idx, ni*W + nj)

    # Count the number of distinct connected components among green cells
    # i.e. count distinct parents among green cells
    comp_set = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                comp_set.add(find(i*W + j))
    c = len(comp_set)  # number of green connected components

    # Now, let's compute the contribution for each red cell
    # If we paint a red cell green, the new number of connected components = c + 1 - k
    # where k is the number of distinct green components among its (up to) 4 neighbors.
    # (If a cell is not adjacent to any green cell, k=0, so the new total is c+1.
    #  Otherwise, it merges all those k distinct components into 1, so we reduce (k-1).)
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))

    # The total number of red cells
    R = len(red_cells)

    # Sum over all red cells [c + 1 - k]
    total_val = 0
    for (i, j) in red_cells:
        # find distinct parent components among neighbors that are green
        neighbor_parents = set()
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                neighbor_parents.add(find(ni*W + nj))
        k = len(neighbor_parents)
        total_val += (c + 1 - k)

    # We want the expected value = total_val / R, in mod 998244353
    # Let mod = 998244353. We'll compute total_val * inv(R) mod 998244353
    mod = 998244353

    # Function to compute modular inverse using Fermat's little theorem (mod is prime)
    def mod_inv(x, m=mod):
        return pow(x, m-2, m)

    # We do total_val mod 998244353
    total_val %= mod
    inv_R = mod_inv(R)
    ans = (total_val * inv_R) % mod

    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()