def main():
    import sys
    from collections import deque
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    grid = [list(next(it).strip()) for _ in range(H)]
    
    # Precompute which cells are "safe". A safe cell is an empty cell ('.')
    # that does not have any magnet ('#') in its four neighbors.
    safe = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '.':
                continue
            # Check the four adjacent cells.
            is_safe = True
            if i > 0 and grid[i-1][j] == '#':
                is_safe = False
            if i < H-1 and grid[i+1][j] == '#':
                is_safe = False
            if j > 0 and grid[i][j-1] == '#':
                is_safe = False
            if j < W-1 and grid[i][j+1] == '#':
                is_safe = False
            safe[i][j] = is_safe

    # For movement:
    # - If you are on a cell that has an adjacent magnet, you cannot move at all.
    # - Otherwise, from a safe cell you can move (in one step) to any adjacent empty cell.
    #
    # Note:
    #  - Even if the destination cell is "unsafe" (i.e. adjacent to a magnet),
    #    it can still be reached from a safe cell.
    #  - However, if you ever find yourself in an unsafe cell, you are stuck there.
    #
    # Therefore, for a safe cell starting position, you can reach:
    #   • All safe cells in its safe-connected component (cells which you can reach 
    #     from safe-to-safe moves).
    #   • And any adjacent unsafe empty cell (since you can move from a safe cell 
    #     directly into an unsafe one).
    #
    # For any unsafe cell, the degree of freedom is just 1 (the cell itself).
    
    # We'll use BFS to compute connected components of safe cells.
    visited = [[False]*W for _ in range(H)]
    # To mark the unsafe cells that have been added as neighbors to a safe component.
    unsafe_marked = [[False]*W for _ in range(H)]
    
    max_degree = 1  # At worst, an empty unsafe cell yields degree 1.
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for i in range(H):
        for j in range(W):
            # Process only safe empty cells that are not visited yet.
            if grid[i][j] != '.' or not safe[i][j] or visited[i][j]:
                continue
            dq = deque()
            dq.append((i, j))
            visited[i][j] = True
            safe_count = 0
            adjacent_unsafe = 0
            while dq:
                x, y = dq.popleft()
                safe_count += 1
                # Check all four neighbors.
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < H and 0 <= ny < W):
                        continue
                    if grid[nx][ny] != '.':
                        # Cannot move into a magnet cell.
                        continue
                    if safe[nx][ny]:
                        # If neighbor is safe and not visited, add it to the component.
                        if not visited[nx][ny]:
                            visited[nx][ny] = True
                            dq.append((nx, ny))
                    else:
                        # This neighbor is an empty but unsafe cell.
                        if not unsafe_marked[nx][ny]:
                            unsafe_marked[nx][ny] = True
                            adjacent_unsafe += 1
            # The degree (reachable cells count) for any cell in this safe component is:
            #   number of safe cells in the component + number of adjacent unsafe empty cells.
            degree = safe_count + adjacent_unsafe
            if degree > max_degree:
                max_degree = degree
    
    sys.stdout.write(str(max_degree))
    
if __name__ == '__main__':
    main()