import heapq
import bisect

def main():
    import sys
    H, W, Y = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        row = list(map(int, sys.stdin.readline().split()))
        A.append(row)
    
    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    heap = []
    
    # Initialize edge cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                val = A[i][j]
                if dist[i][j] > val:
                    dist[i][j] = val
                    heapq.heappush(heap, (val, i, j))
    
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while heap:
        current_max, i, j = heapq.heappop(heap)
        if current_max > dist[i][j]:
            continue
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                new_max = max(current_max, A[ni][nj])
                if new_max < dist[ni][nj]:
                    dist[ni][nj] = new_max
                    heapq.heappush(heap, (new_max, ni, nj))
    
    times = []
    for i in range(H):
        for j in range(W):
            if dist[i][j] != INF:
                times.append(dist[i][j])
    
    times.sort()
    
    for i in range(1, Y+1):
        cnt = bisect.bisect_right(times, i)
        remaining = H * W - cnt
        print(remaining)
    
if __name__ == "__main__":
    main()