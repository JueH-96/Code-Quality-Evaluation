import heapq

def main():
    import sys
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    start = (A-1, B-1)
    goal = (C-1, D-1)
    
    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    dist[start[0]][start[1]] = 0
    heap = []
    heapq.heappush(heap, (0, start[0], start[1]))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while heap:
        current_k, i, j = heapq.heappop(heap)
        if (i, j) == goal:
            print(current_k)
            return
        if current_k > dist[i][j]:
            continue
        for dx, dy in directions:
            for s in range(1, 3):  # s is 1 or 2
                ni = i + dx * s
                nj = j + dy * s
                if 0 <= ni < H and 0 <= nj < W:
                    has_wall = False
                    for step in range(1, s + 1):
                        x = i + dx * step
                        y = j + dy * step
                        if grid[x][y] == '#':
                            has_wall = True
                    new_k = current_k + (1 if has_wall else 0)
                    if new_k < dist[ni][nj]:
                        dist[ni][nj] = new_k
                        heapq.heappush(heap, (new_k, ni, nj))
    
if __name__ == "__main__":
    main()