import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # visited[r][c] == 1  -> the cell is already humidified (counted)
    visited = [bytearray(W) for _ in range(H)]
    q = deque()               # store cells as a single integer  r * W + c

    # enqueue all humidifiers
    for r in range(H):
        row = grid[r]
        for c, ch in enumerate(row):
            if ch == 'H':
                visited[r][c] = 1
                q.append(r * W + c)

    # If D == 0 we do not need to expand
    if D:
        # BFS layer by layer up to depth D
        for _depth in range(D):
            if not q:                     # frontier empty → nothing more to spread
                break
            for _ in range(len(q)):       # process current layer completely
                idx = q.popleft()
                r, c = divmod(idx, W)
                # four neighbours
                if r and not visited[r-1][c] and grid[r-1][c] != '#':
                    visited[r-1][c] = 1
                    q.append((r-1) * W + c)
                if r + 1 < H and not visited[r+1][c] and grid[r+1][c] != '#':
                    visited[r+1][c] = 1
                    q.append((r+1) * W + c)
                if c and not visited[r][c-1] and grid[r][c-1] != '#':
                    visited[r][c-1] = 1
                    q.append(r * W + (c-1))
                if c + 1 < W and not visited[r][c+1] and grid[r][c+1] != '#':
                    visited[r][c+1] = 1
                    q.append(r * W + (c+1))

    # Count humidified floor cells (visited never marks walls)
    answer = sum(sum(row) for row in visited)
    print(answer)

if __name__ == "__main__":
    main()