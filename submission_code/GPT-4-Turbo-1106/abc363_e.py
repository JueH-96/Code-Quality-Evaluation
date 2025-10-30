from collections import deque

def read_ints():
    return map(int, input().split())

def bfs(sink_map, queue, sea_level):
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not sink_map[nx][ny] and A[nx][ny] <= sea_level:
                sink_map[nx][ny] = True
                queue.append((nx, ny))

def simulate(H, W, Y, A):
    sink_map = [[False] * W for _ in range(H)]
    queue = deque()
    
    # Initialize the queue with border sections
    for i in range(H):
        if A[i][0] == 0:
            sink_map[i][0] = True
            queue.append((i, 0))
        if A[i][W-1] == 0:
            sink_map[i][W-1] = True
            queue.append((i, W-1))
    for j in range(W):
        if A[0][j] == 0:
            sink_map[0][j] = True
            queue.append((0, j))
        if A[H-1][j] == 0:
            sink_map[H-1][j] = True
            queue.append((H-1, j))
    
    # Perform BFS for initial sinking sections
    bfs(sink_map, queue, 0)
    
    # Simulate each year
    for year in range(1, Y + 1):
        # Increase sea level and check for new sinking sections
        for i in range(H):
            for j in range(W):
                if not sink_map[i][j] and A[i][j] <= year:
                    sink_map[i][j] = True
                    queue.append((i, j))
        bfs(sink_map, queue, year)
        
        # Count remaining sections
        remaining_area = sum(not sunk for row in sink_map for sunk in row)
        print(H * W - remaining_area)

# Read input
H, W, Y = read_ints()
A = [list(read_ints()) for _ in range(H)]

simulate(H, W, Y, A)