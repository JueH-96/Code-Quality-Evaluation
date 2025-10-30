import heapq

H, W, Y = map(int, input().split())
A = []
for _ in range(H):
    row = list(map(int, input().split()))
    A.append(row)

# Compute the earliest year each cell will sink
sink_year = [[float('inf')] * W for _ in range(H)]
heap = []

# Add border cells
for i in range(H):
    for j in range(W):
        if i == 0 or i == H-1 or j == 0 or j == W-1:
            sink_year[i][j] = A[i][j]
            heapq.heappush(heap, (A[i][j], i, j))

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

while heap:
    current_year, x, y = heapq.heappop(heap)
    
    if current_year > sink_year[x][y]:
        continue
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W:
            new_year = max(A[nx][ny], current_year)
            if new_year < sink_year[nx][ny]:
                sink_year[nx][ny] = new_year
                heapq.heappush(heap, (new_year, nx, ny))

# For each year, count how many cells remain
for year in range(1, Y + 1):
    remaining = 0
    for i in range(H):
        for j in range(W):
            if sink_year[i][j] > year:
                remaining += 1
    print(remaining)