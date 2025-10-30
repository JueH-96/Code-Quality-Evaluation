def main():
    import sys
    from collections import deque

    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return
    # Parse H and W.
    H, W = map(int, data[0].split())
    grid = [data[i].decode() for i in range(1, H+1)]
    # Parse starting and goal positions.
    A, B, C, D = map(int, data[H+1].split())
    A, B, C, D = A-1, B-1, C-1, D-1  # use 0-index

    # We want to compute the minimum number of front kicks needed.
    # There are two kinds of moves:
    #   1. Adjacent moves (up,down,left,right) which cost 0.
    #      However, adjacent moves are only allowed if the target cell is a road.
    #   2. A "front kick" move. When at cell (x,y) with cost c, if you front kick in one of the 4 directions,
    #      every cell which is exactly 1 or 2 steps away in that direction becomes a road.
    #      Although you remain in the same cell, after the kick these cells are now usable.
    #      We can then (in future 0‐cost adjacent moves) move into those cells.
    #      The cost of performing a front kick is 1.
    #
    # We use a 0‐1 BFS. Let dist[i][j] be the minimum front kicks needed for cell (i,j) to be reached.
    # Initially only those cells that are originally road ('.') are accessible via adjacent moves.
    # The bomb (front kick) action may convert a wall cell into a road at an additional cost.
    
    INF = 10**9
    dist = [[INF] * W for _ in range(H)]
    dq = deque()
    dist[A][B] = 0
    dq.append((A, B))
    
    # Directions for up, down, left, right.
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while dq:
        x, y = dq.popleft()
        current_cost = dist[x][y]
        # If we've reached the destination, output the cost.
        if x == C and y == D:
            sys.stdout.write(str(current_cost))
            return
        
        # Adjacent moves – cost 0.
        # Only cells originally road ('.') can be moved into (unless it was previously converted by a bomb,
        # in which case our dist[][] value would already be updated).
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] == '.' and dist[nx][ny] > current_cost:
                    dist[nx][ny] = current_cost
                    dq.appendleft((nx, ny))
        
        # Front kick moves – cost +1.
        # When performing a front kick, for each chosen direction and for distance 1 and 2,
        # the target cell becomes road (if within grid) regardless of its original state.
        # We update its cost to current_cost + 1 if we get a better (lower) value.
        new_cost = current_cost + 1
        for dx, dy in moves:
            # d represents the distance (1 or 2 steps away in that direction)
            for d in (1, 2):
                nx, ny = x + dx * d, y + dy * d
                if not (0 <= nx < H and 0 <= ny < W):
                    # If the cell is out-of-bounds, then cells further in this direction will be out as well.
                    break
                if dist[nx][ny] > new_cost:
                    dist[nx][ny] = new_cost
                    dq.append((nx, ny))
    # If the destination is not reached, output -1.
    sys.stdout.write("-1")
    
if __name__ == '__main__':
    main()