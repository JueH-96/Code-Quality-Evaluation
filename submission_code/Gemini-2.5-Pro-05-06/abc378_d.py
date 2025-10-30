def solve():
    H, W, K = map(int, input().split())
    grid = [input() for _ in range(H)]

    ans = 0

    # dr, dc define relative coordinates for neighbors: Up, Down, Left, Right
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # DFS function:
    # r, c: current cell coordinates
    # k_left: number of moves remaining to make
    # visited: a set of (row, col) tuples of cells already in the current path
    def dfs(r, c, k_left, visited):
        # Base case: if K moves have been made (k_left is 0)
        if k_left == 0:
            return 1  # This path is complete and valid

        path_count_from_here = 0
        # Explore neighbors
        for i in range(4):  # Iterate through 4 directions
            nr, nc = r + dr[i], c + dc[i]

            # Check if the neighbor is valid
            if 0 <= nr < H and 0 <= nc < W:      # Check if within grid boundaries
                if grid[nr][nc] == '.':          # Check if cell is empty
                    if (nr, nc) not in visited:  # Check if cell not visited in current path
                        # If valid, add to visited set, recurse, then backtrack
                        visited.add((nr, nc))
                        path_count_from_here += dfs(nr, nc, k_left - 1, visited)
                        visited.remove((nr, nc)) # Backtrack for other explorations
        
        return path_count_from_here

    # Iterate over all cells in the grid to find possible starting points
    for r_start in range(H):
        for c_start in range(W):
            if grid[r_start][c_start] == '.':  # A starting cell must be empty
                # Initialize visited set for a new path starting at (r_start, c_start)
                # The starting cell itself is part of the path and is considered visited.
                visited_path = set()
                visited_path.add((r_start, c_start))
                
                # Call DFS. We need to make K moves from (r_start, c_start).
                ans += dfs(r_start, c_start, K, visited_path)
                
                # No need to remove (r_start, c_start) from visited_path here explicitly,
                # as visited_path is local to this loop iteration. A new set is created
                # for the next potential starting cell.
    
    print(ans)

if __name__ == '__main__':
    solve()