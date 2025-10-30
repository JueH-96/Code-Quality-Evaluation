# YOUR CODE HERE
def solve():
    """
    Reads input, initializes state, and calls the recursive solver.
    Prints "Yes" or "No" based on the result.
    """
    
    # It is not necessary to increase the recursion limit for N<=7.
    # The default limit is usually 1000, and the max depth here is N.
    # import sys
    # sys.setrecursionlimit(10**6) 

    N, H, W = map(int, input().split())
    tiles = [tuple(map(int, input().split())) for _ in range(N)]

    # Heuristic: Try placing larger tiles first. This can significantly
    # prune the search tree by making more impactful decisions earlier.
    tiles.sort(key=lambda t: t[0] * t[1], reverse=True)

    # grid[r][c] is True if the cell is covered, False otherwise.
    grid = [[False] * W for _ in range(H)]
    # used[i] is True if the i-th tile has been placed.
    used = [False] * N

    def can_tile():
        """
        Recursively tries to tile the grid using a backtracking approach.
        The state is implicitly defined by the `grid` and `used` arrays
        from the enclosing scope.
        Returns True if a valid tiling is found, False otherwise.
        """
        
        # 1. Find the first uncovered cell (top-to-bottom, left-to-right).
        # This canonical search order prunes the search space by avoiding
        # exploring permutations of placement order that lead to the same grid state.
        first_r, first_c = -1, -1
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    first_r, first_c = r, c
                    break
            if first_r != -1:
                break

        # 2. Base Case: If no uncovered cell is found, the grid is perfectly tiled.
        if first_r == -1:
            return True

        # 3. Recursive Step: Try to place each available tile at the first empty cell.
        for i in range(N):
            if used[i]:
                continue

            a, b = tiles[i]
            
            # Create a list of orientations to try for the current tile.
            # If the tile is a square, we only need to check one orientation.
            orientations = [(a, b)]
            if a != b:
                orientations.append((b, a))

            for h, w in orientations:
                # Check if the tile fits within the grid boundaries.
                if first_r + h > H or first_c + w > W:
                    continue

                # Check if the area required for the tile is currently empty.
                can_place = True
                for r_offset in range(h):
                    for c_offset in range(w):
                        if grid[first_r + r_offset][first_c + c_offset]:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # If it can be placed, update the state (place the tile).
                    used[i] = True
                    for r_offset in range(h):
                        for c_offset in range(w):
                            grid[first_r + r_offset][first_c + c_offset] = True
                    
                    # Recurse with the new state.
                    if can_tile():
                        # If the recursive call found a solution, propagate success.
                        return True
                    
                    # Backtrack: If the recursive call failed, undo the move.
                    used[i] = False
                    for r_offset in range(h):
                        for c_offset in range(w):
                            grid[first_r + r_offset][first_c + c_offset] = False

        # 4. If we tried all tiles at this position and none led to a solution, backtrack.
        return False

    if can_tile():
        print("Yes")
    else:
        print("No")

solve()