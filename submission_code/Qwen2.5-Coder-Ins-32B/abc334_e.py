# YOUR CODE HERE
MOD = 998244353

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1
        return True
    return False

def count_components(grid, H, W):
    parent = {}
    rank = {}
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                parent[(i, j)] = (i, j)
                rank[(i, j)] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        union(parent, rank, (i, j), (ni, nj))

    components = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                components.add(find(parent, (i, j)))
    return len(components)

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    grid = input[2:]

    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))

    initial_components = count_components(grid, H, W)
    total_components = 0

    for ri, rj in red_cells:
        new_grid = [list(row) for row in grid]
        new_grid[ri][rj] = '#'
        new_components = count_components(new_grid, H, W)
        total_components += new_components

    expected_value = total_components * mod_inverse(len(red_cells), MOD)
    print(expected_value % MOD)

if __name__ == "__main__":
    main()