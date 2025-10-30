def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [list(input().rstrip("
")) for _ in range(H)]
    A, B, C, D = map(int, input().split())
    # convert to 0-indexed coordinates
    A, B, C, D = A-1, B-1, C-1, D-1

    # We'll keep a cost matrix storing the minimum front kicks needed to "open" and reach each cell.
    INF = 10**9
    cost = [[INF] * W for _ in range(H)]
    dq = deque()
    cost[A][B] = 0
    dq.append((A, B))
    
    # Standard 4 directions: up, down, left, right.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # We use a 0-1 BFS where moving to an adjacent road cell costs 0 and a "front kick" costs 1.
    # Note: When a front kick is performed, it affects the cells exactly 1 and 2 steps away in one chosen direction.
    # If any of those cells are walls, they are turned into roads permanently (we update grid accordingly).
    while dq:
        i, j = dq.popleft()
        c = cost[i][j]
        if i == C and j == D:
            print(c)
            return
        
        # Free moves: move to adjacent cell if it is already a road.
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and cost[ni][nj] > c:
                    cost[ni][nj] = c
                    dq.appendleft((ni, nj))
        
        # Front kicks (conversion moves): cost increases by 1.
        # For each direction, the kick converts the cell 1 step and 2 steps away.
        nc = c + 1
        for di, dj in directions:
            for step in (1, 2):
                ni, nj = i + di * step, j + dj * step
                if 0 <= ni < H and 0 <= nj < W:
                    if cost[ni][nj] > nc:
                        cost[ni][nj] = nc
                        # If the cell is a wall, the kick converts it into a road.
                        if grid[ni][nj] == '#':
                            grid[ni][nj] = '.'
                        dq.append((ni, nj))
                        
    print(-1)
    
if __name__ == '__main__':
    main()