def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    H = int(input_data[0])
    W = int(input_data[1])
    grid = input_data[2:]
    
    # Create a 2D boolean array to mark free cells (cell is free if it has '.')
    free_cell = [[(grid[i][j] == '.') for j in range(W)] for i in range(H)]
    
    # For each free cell, determine if the cell is "safe" – i.e., no magnet ('#')
    # appears in a vertically or horizontally adjacent cell.
    safe = [[False] * W for _ in range(H)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(H):
        for j in range(W):
            if not free_cell[i][j]:
                continue
            is_safe = True
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '#':
                        is_safe = False
                        break
            safe[i][j] = is_safe

    # We now simulate the possible moves.
    # Note: The movement rule is that from a cell you can only move if that cell is safe.
    # Even if a cell is reached but is unsafe, it is counted in the reachable set;
    # however, you cannot move further from it.
    #
    # Thus, to compute the "degree of freedom" (the number of cells reachable from a 
    # starting cell following these rules), we perform a search where:
    #   - When at a safe cell, we try to move to all adjacent free cells that have not been visited.
    #   - When at an unsafe cell, we are stuck so we do not expand further.
    #
    # We will compute connected components using BFS starting from every safe cell
    # that has not yet been visited. This procedure finds the maximum reachable set from any
    # safe cell. 
    # Note: A free cell that is unsafe and never reached (i.e. not connected to any safe cell)
    # always has a degree of freedom equal to 1.
    
    visited = [[False] * W for _ in range(H)]
    max_degree = 0

    for i in range(H):
        for j in range(W):
            if free_cell[i][j] and safe[i][j] and not visited[i][j]:
                component_size = 0
                dq = deque()
                dq.append((i, j))
                visited[i][j] = True
                while dq:
                    x, y = dq.popleft()
                    component_size += 1
                    # We can only move from a cell if it is safe.
                    if safe[x][y]:
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if free_cell[nx][ny] and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    dq.append((nx, ny))
                max_degree = max(max_degree, component_size)

    # Finally, some free cells might never have been reached by a search from any safe cell.
    # Those cells are unsafe, so from each such cell Takahashi can't move at all (degree 1).
    for i in range(H):
        for j in range(W):
            if free_cell[i][j] and not visited[i][j]:
                max_degree = max(max_degree, 1)

    print(max_degree)


if __name__ == "__main__":
    main()