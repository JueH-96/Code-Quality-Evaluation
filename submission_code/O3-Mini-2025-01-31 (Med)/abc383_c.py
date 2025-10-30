def main():
    import sys
    from collections import deque

    data = sys.stdin.read().splitlines()
    if not data:
        return

    H, W, D = map(int, data[0].split())
    grid = [list(line.strip()) for line in data[1:H+1]]
    
    # Prepare a 2D list to record distance from the closest humidifier, -1 means unvisited.
    dist = [[-1] * W for _ in range(H)]
    q = deque()
    
    # Initialize multi-source queue for all humidifier positions.
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                q.append((i, j))
                dist[i][j] = 0  # Humidifier cells have 0 distance.
                
    # Standard BFS (with level limit up to D)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        i, j = q.popleft()
        current_distance = dist[i][j]
        # Only explore further if we haven't exceeded D moves.
        if current_distance == D:
            continue
        for di, dj in directions:
            ni, nj = i + di, j + dj
            # Check boundaries and wall conditions.
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if grid[ni][nj] == '#':
                continue
            # If the cell is not visited and within allowed moves, mark it and push to queue.
            if dist[ni][nj] == -1:
                nd = current_distance + 1
                if nd <= D:
                    dist[ni][nj] = nd
                    q.append((ni, nj))
                    
    # Count and output the number of humidified floor cells (cells not containing a wall and visited).
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1:
                count += 1

    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()