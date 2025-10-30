def expected_green_components(H, W, grid):
    MOD = 998244353

    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Function to perform DFS and find connected components of green cells
    def dfs(x, y, visited):
        stack = [(x, y)]
        visited.add((x, y))
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] == '#':
                    visited.add((nx, ny))
                    stack.append((nx, ny))

    # Count initial green components
    visited = set()
    initial_components = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and (i, j) not in visited:
                dfs(i, j, visited)
                initial_components += 1

    # Count red cells and potential new components
    red_cells = []
    expected_value = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Check how many components would be affected if this red cell is turned green
                adjacent_components = set()
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                        adjacent_components.add((nx, ny))

                # If no adjacent green cells, it creates a new component
                if not adjacent_components:
                    new_components = initial_components + 1
                else:
                    # Count how many unique components are adjacent
                    new_components = initial_components - len(adjacent_components) + 1

                expected_value += new_components

                # Store red cell for later use
                red_cells.append((i, j))

    # Calculate the expected value as a fraction
    num_red_cells = len(red_cells)
    if num_red_cells == 0:
        return 0

    # Expected value is the sum of new components divided by the number of red cells
    P = expected_value
    Q = num_red_cells

    # To find R such that R * Q ≡ P (mod MOD), we need the modular inverse of Q
    def mod_inverse(a, m):
        return pow(a, m - 2, m)

    Q_inv = mod_inverse(Q, MOD)
    R = (P * Q_inv) % MOD

    return R

import sys
input = sys.stdin.read
data = input().splitlines()
H, W = map(int, data[0].split())
grid = [list(data[i + 1]) for i in range(H)]

result = expected_green_components(H, W, grid)
print(result)