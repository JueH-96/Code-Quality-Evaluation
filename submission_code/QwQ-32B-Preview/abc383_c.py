from collections import deque

def main():
    import sys
    from itertools import product

    # Read all input at once
    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = [list(data[i + 3]) for i in range(H)]

    # Define directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize distance grid
    dist = [[2000 for _ in range(W)] for _ in range(H)]
    queue = deque()

    # Collect all humidifier positions and set their distance to 0
    for i, j in product(range(H), range(W)):
        if grid[i][j] == 'H':
            dist[i][j] = 0
            queue.append((i, j))
        elif grid[i][j] == '#':
            dist[i][j] = -1  # Mark walls as invalid

    # Perform BFS
    while queue:
        i, j = queue.popleft()
        current_dist = dist[i][j]
        if current_dist >= D:
            continue  # No need to explore further if distance exceeds D
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and dist[ni][nj] == 2000:
                    dist[ni][nj] = current_dist + 1
                    queue.append((ni, nj))

    # Count the number of floor cells within distance D
    count = 0
    for i, j in product(range(H), range(W)):
        if grid[i][j] == '.' and dist[i][j] <= D:
            count += 1
    print(count)

if __name__ == "__main__":
    main()