import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().splitlines()
    H, W = map(int, input_data[0].split())
    grid = input_data[1:]

    pattern = "snuke"

    # Start cell must be 's'
    if grid[0][0] != 's':
        print("No")
        return

    # visited[row][col][idx]  idx = position in pattern expected at this cell
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
    q = deque()
    visited[0][0][0] = True
    q.append((0, 0, 0))                 # row, col, idx (0 means 's')

    drc = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c, idx = q.popleft()
        if r == H - 1 and c == W - 1:
            print("Yes")
            return
        next_idx = (idx + 1) % 5
        need = pattern[next_idx]
        for dr, dc in drc:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr][nc][next_idx] and grid[nr][nc] == need:
                    visited[nr][nc][next_idx] = True
                    q.append((nr, nc, next_idx))

    print("No")


if __name__ == "__main__":
    main()