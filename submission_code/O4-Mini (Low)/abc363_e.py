import sys
import threading
import heapq

def main():
    import sys
    input = sys.stdin.readline
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    N = H * W

    # Directions
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    # t[i][j]: the minimum water level at which cell (i,j) is flooded
    # Initialize all to a large value.
    INF = 10**9
    t = [[INF]*W for _ in range(H)]
    pq = []

    # Multi-source Dijkstra: boundary cells start with t = A[i][j]
    for i in range(H):
        for j in (0, W-1):
            t[i][j] = A[i][j]
            heapq.heappush(pq, (t[i][j], i, j))
    for j in range(W):
        for i in (0, H-1):
            if t[i][j] > A[i][j]:
                t[i][j] = A[i][j]
                heapq.heappush(pq, (t[i][j], i, j))

    # Dijkstra: for each cell, propagation cost is max(current path cost, neighbor elevation)
    while pq:
        cur_t, x, y = heapq.heappop(pq)
        if cur_t > t[x][y]:
            continue
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                nt = max(cur_t, A[nx][ny])
                if nt < t[nx][ny]:
                    t[nx][ny] = nt
                    heapq.heappush(pq, (nt, nx, ny))

    # Build frequency of flood times up to Y
    # freq[k] = number of cells with t == k, for k <= Y
    freq = [0] * (Y + 2)
    for i in range(H):
        for j in range(W):
            if t[i][j] <= Y:
                freq[t[i][j]] += 1
            # if t>Y, these never flood in our query range

    # prefix sum of flooded cells by year
    for year in range(1, Y+1):
        freq[year] += freq[year-1]

    out = []
    for year in range(1, Y+1):
        flooded = freq[year]
        remaining = N - flooded
        out.append(str(remaining))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()