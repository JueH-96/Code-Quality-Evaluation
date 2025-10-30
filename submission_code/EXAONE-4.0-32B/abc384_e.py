import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    X = int(next(it))
    P = int(next(it))
    Q = int(next(it))
    grid = []
    for _ in range(H):
        row = list(map(int, [next(it) for _ in range(W)]))
        grid.append(row)
    
    start_i = P - 1
    start_j = Q - 1
    total = grid[start_i][start_j]
    visited = [[False] * W for _ in range(H)]
    heap = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    visited[start_i][start_j] = True
    
    for dx, dy in directions:
        ni, nj = start_i + dx, start_j + dy
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            thresh = X * grid[ni][nj] + 1
            heapq.heappush(heap, (thresh, ni, nj))
    
    while heap:
        thresh, i, j = heapq.heappop(heap)
        if visited[i][j]:
            continue
        if total >= thresh:
            total += grid[i][j]
            visited[i][j] = True
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                    thresh_new = X * grid[ni][nj] + 1
                    heapq.heappush(heap, (thresh_new, ni, nj))
        else:
            heapq.heappush(heap, (thresh, i, j))
            break
    
    print(total)

if __name__ == "__main__":
    main()