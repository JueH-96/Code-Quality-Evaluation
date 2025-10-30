import sys
import threading

def main():
    import sys
    import heapq

    data = sys.stdin.read().split()
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    X = int(next(it))
    P = int(next(it)) - 1
    Q = int(next(it)) - 1

    # Read grid
    S = [list(map(int, (next(it) for _ in range(W)))) for _ in range(H)]

    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # visited: absorbed cells
    visited = [[False]*W for _ in range(H)]
    # in_queue: cells already pushed into frontier heap
    in_queue = [[False]*W for _ in range(H)]

    # Min-heap of (weight, i, j)
    heap = []

    # Initial position
    cur = S[P][Q]
    visited[P][Q] = True

    # Push initial neighbors
    for di, dj in dirs:
        ni, nj = P + di, Q + dj
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            heapq.heappush(heap, (S[ni][nj], ni, nj))
            in_queue[ni][nj] = True

    # Greedily absorb smallest eligible slime
    # Condition: weight < cur/X  <=> weight*X < cur
    # To avoid overflow, test weight <= (cur-1)//X
    while heap:
        w, i, j = heap[0]
        # max allowed weight = floor((cur-1)/X)
        max_w = (cur - 1) // X
        if w > max_w:
            break
        # pop and absorb
        heapq.heappop(heap)
        cur += w
        visited[i][j] = True
        # push its neighbors
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and not in_queue[ni][nj]:
                heapq.heappush(heap, (S[ni][nj], ni, nj))
                in_queue[ni][nj] = True

    # Output result
    print(cur)

if __name__ == "__main__":
    main()