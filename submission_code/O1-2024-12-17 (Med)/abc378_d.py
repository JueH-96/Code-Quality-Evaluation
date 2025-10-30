def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    
    # Parse inputs
    H, W, K = map(int, input_data[:3])
    grid_chars = input_data[3:]
    
    # Build grid
    grid = [grid_chars[r] for r in range(H)]
    
    # Create a list of valid neighbors (adjacency) for each cell
    # We'll index cells by index = r * W + c
    adjacency = [[] for _ in range(H * W)]
    def idx(r, c):
        return r * W + c
    
    # Precompute which cells are empty and build adjacency for them
    empty_cells = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                empty_cells.append(idx(r, c))
    
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                continue
            cur_index = idx(r, c)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '.':
                        adjacency[cur_index].append(idx(nr, nc))
    
    visited = [False] * (H * W)
    answer = 0
    
    # Depth-first search with backtracking
    def dfs(pos, depth):
        nonlocal answer
        if depth == K:
            # We used exactly K moves, so the path length is K+1
            answer += 1
            return
        for nxt in adjacency[pos]:
            if not visited[nxt]:
                visited[nxt] = True
                dfs(nxt, depth + 1)
                visited[nxt] = False
    
    # Try each empty cell as a starting point
    for start in empty_cells:
        visited[start] = True
        dfs(start, 0)
        visited[start] = False
    
    # Print the result
    print(answer)

# Call main() to execute the solution.
if __name__ == "__main__":
    main()