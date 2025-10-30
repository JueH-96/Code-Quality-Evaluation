def solve():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    P -= 1  # Convert to 0-indexed
    Q -= 1
    
    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W
    
    max_strength = [grid[P][Q]]
    
    def dfs(r, c, strength, absorbed):
        max_strength[0] = max(max_strength[0], strength)
        
        # Get adjacent cells that can be absorbed
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and (nr, nc) not in absorbed:
                # Can absorb if slime_strength * X < current_strength
                if grid[nr][nc] * X < strength:
                    absorbed.add((nr, nc))
                    dfs(nr, nc, strength + grid[nr][nc], absorbed)
                    absorbed.remove((nr, nc))  # backtrack
    
    initial_absorbed = {(P, Q)}
    dfs(P, Q, grid[P][Q], initial_absorbed)
    print(max_strength[0])

solve()