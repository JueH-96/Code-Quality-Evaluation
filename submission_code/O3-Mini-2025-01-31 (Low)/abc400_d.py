def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    A, B, C, D = map(int, input().split())
    # Convert to 0-indexed coordinates.
    A, B, C, D = A - 1, B - 1, C - 1, D - 1

    # dp[i][j] will hold the minimum number of front kicks needed to reach cell (i, j).
    INF = 10**9
    dp = [[INF] * W for _ in range(H)]
    
    dq = deque()
    dp[A][B] = 0
    dq.append((A, B))
    
    # We will perform a 0-1 BFS.
    # Adjacent moves (up,down,left,right) are free (cost 0).
    # A front kick when performed at (x,y) in any of the 4 directions costs +1.
    # When a front kick is performed, for that direction, the cells 1 step and 2 steps away (if in bounds)
    # become roads (even if they were walls).
    # Note that even if a cell is already road, we can use a front kick to "clear" a cell further away.
    # Once a wall is cleared (turned into a road) it stays road.
    
    adjacent_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while dq:
        x, y = dq.popleft()
        current_cost = dp[x][y]
        
        # Early exit: since 0-1 BFS always pops the smallest cost first, if we reached our goal we can output.
        if x == C and y == D:
            print(current_cost)
            return
        
        # 0 cost moves: move into adjacent cells if they are roads.
        for dx, dy in adjacent_dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] == '.' and dp[nx][ny] > current_cost:
                    dp[nx][ny] = current_cost
                    dq.appendleft((nx, ny))
        
        # 1 cost moves: perform a front kick in all 4 directions.
        # For each direction, check the cells 1 and 2 steps away.
        for dx, dy in adjacent_dirs:
            for d in range(1, 3):
                nx, ny = x + dx * d, y + dy * d
                if not (0 <= nx < H and 0 <= ny < W):
                    break  # out-of-bound; further cells in this direction are also out of bounds.
                if dp[nx][ny] <= current_cost + 1:
                    continue  # already reached with an equal or lower cost.
                dp[nx][ny] = current_cost + 1
                # Mark cell as road regardless of whether it was originally road or a wall.
                grid[nx][ny] = '.'
                dq.append((nx, ny))
    
    # If unreachable (should not occur as problem guarantees connectivity after kicks), print -1.
    print(-1)


if __name__ == "__main__":
    main()