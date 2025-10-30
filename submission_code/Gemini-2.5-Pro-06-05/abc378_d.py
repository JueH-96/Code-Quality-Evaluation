import sys

def main():
    """
    This program solves the grid path counting problem.
    It reads grid dimensions, the number of moves K, and the grid layout.
    It then uses a backtracking algorithm (recursive DFS) to count all valid paths.
    A valid path starts at any empty cell, consists of K moves to adjacent,
    unblocked cells, and does not visit any cell more than once.
    """

    # For deep recursion, it's a good practice to increase the recursion limit.
    # The default is often 1000. Max depth here is K+1=12, so it's not strictly
    # necessary but included as a safeguard.
    sys.setrecursionlimit(2000)

    try:
        # Read problem parameters from standard input
        H, W, K = map(int, sys.stdin.readline().split())
        # Read the grid
        grid = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Exit gracefully if input is malformed or empty
        return

    # Directions for adjacent moves: up, down, left, right
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    def dfs(r, c, moves_made, visited):
        """
        Recursively explores paths from the current cell (r, c) using DFS with backtracking.
        
        Args:
            r (int): Current row index.
            c (int): Current column index.
            moves_made (int): The number of moves made to reach this cell.
            visited (set): A set of (row, col) tuples for cells in the current path.
        
        Returns:
            int: The number of valid paths of length K that can be formed by extending
                 the current path.
        """
        # Base case: If we have made exactly K moves, we have found one complete path.
        if moves_made == K:
            return 1

        path_count = 0
        # Explore all four adjacent cells
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check if the next move is valid:
            # 1. Within grid boundaries.
            # 2. To an empty cell ('.').
            # 3. To a cell not yet visited in the current path.
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and (nr, nc) not in visited:
                # If valid, add the new cell to the visited set and recurse
                visited.add((nr, nc))
                path_count += dfs(nr, nc, moves_made + 1, visited)
                # Backtrack: remove the cell to allow it to be part of other valid paths
                visited.remove((nr, nc))
        
        return path_count

    total_paths = 0
    # Iterate over every cell to consider it as a potential starting point
    for r in range(H):
        for c in range(W):
            # A path must start from an empty cell
            if grid[r][c] == '.':
                # Initialize a new visited set for a path starting at (r, c)
                visited = {(r, c)}
                # Start DFS from this cell (0 moves made initially)
                total_paths += dfs(r, c, 0, visited)

    # Print the final total count of valid paths
    print(total_paths)

if __name__ == "__main__":
    main()