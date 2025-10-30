def solve():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    S = []
    for _ in range(H):
        S.append(list(map(int, input().split())))

    P -= 1
    Q -= 1

    def get_neighbors(r, c, grid):
        neighbors = []
        if r > 0:
            neighbors.append((r - 1, c))
        if r < len(grid) - 1:
            neighbors.append((r + 1, c))
        if c > 0:
            neighbors.append((r, c - 1))
        if c < len(grid[0]) - 1:
            neighbors.append((r, c + 1))
        return neighbors

    def can_absorb(strength, neighbor_strength, x):
        return neighbor_strength * x < strength

    def find_max_strength(start_r, start_c, grid, x):
        max_strength = grid[start_r][start_c]
        q = [(grid[start_r][start_c], start_r, start_c, grid)]

        while q:
            curr_strength, curr_r, curr_c, curr_grid = q.pop(0)
            max_strength = max(max_strength, curr_strength)

            neighbors = get_neighbors(curr_r, curr_c, curr_grid)
            
            absorbable_neighbors = []
            for nr, nc in neighbors:
                if curr_grid[nr][nc] != -1 and can_absorb(curr_strength, curr_grid[nr][nc], x):
                    absorbable_neighbors.append((nr, nc))
            
            absorbable_neighbors.sort(key=lambda x: curr_grid[x[0]][x[1]])

            for nr, nc in absorbable_neighbors:
                
                new_grid = [row[:] for row in curr_grid]
                new_strength = curr_strength + new_grid[nr][nc]
                new_grid[nr][nc] = -1
                
                q.append((new_strength, curr_r, curr_c, new_grid))

        return max_strength

    print(find_max_strength(P, Q, S, X))

solve()