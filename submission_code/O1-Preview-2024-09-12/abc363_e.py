# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import heapq
    sys.setrecursionlimit(1 << 25)

    H, W, Y = map(int, sys.stdin.readline().split())
    N = H * W
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    H_indices = range(H)
    W_indices = range(W)

    sink_time = [[float('inf')] * W for _ in range(H)]
    visited = [[False] * W for _ in range(H)]

    import heapq
    heap = []

    # Initialize boundary cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H -1 or j == 0 or j == W -1:
                t = max(1, A[i][j])
                sink_time[i][j] = t
                heapq.heappush(heap, (t, i, j))

    # Dijkstra's Algorithm
    while heap:
        t, x, y = heapq.heappop(heap)
        if visited[x][y]:
            continue
        visited[x][y] = True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                nt = max(t, A[nx][ny])
                if sink_time[nx][ny] > nt:
                    sink_time[nx][ny] = nt
                    heapq.heappush(heap, (nt, nx, ny))

    from collections import defaultdict

    sink_time_list = []
    sink_time_count = defaultdict(int)
    for i in range(H):
        for j in range(W):
            t = sink_time[i][j]
            sink_time_count[t] +=1

    # Build list of events (sorted sink_times)
    events = sorted(sink_time_count.items())
    total_cells = H * W
    area_list = []
    area = total_cells
    idx = 0
    # Convert events to list of (time, area)
    areas_at_time = []
    for t, count in events:
        area -= count
        areas_at_time.append((t, area))

    # Output areas for years from 1 to Y
    output = []
    area = total_cells
    idx = 0
    for year in range(1, Y+1):
        while idx < len(areas_at_time) and areas_at_time[idx][0] == year:
            area = areas_at_time[idx][1]
            idx +=1
        output.append(str(area))
    print('
'.join(output))

threading.Thread(target=main,).start()