from collections import deque
import sys
import threading

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    # The repeating pattern "snuke"
    pattern = "snuke"
    # Check the start cell
    if grid[0][0] != pattern[0]:
        print("No")
        return

    # visited[i][j][m] = have we visited cell (i,j) at step mod 5 == m
    visited = [[[False]*5 for _ in range(W)] for __ in range(H)]
    dq = deque()
    # start at (0,0) with step = 0 (pattern index 0)
    visited[0][0][0] = True
    dq.append((0, 0, 0))

    # 4 directions
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    while dq:
        i, j, m = dq.popleft()
        # If we reached the end, success
        if i == H-1 and j == W-1:
            print("Yes")
            return
        # Next pattern index
        nm = (m + 1) % 5
        need = pattern[nm]
        # Explore neighbors
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if not visited[ni][nj][nm] and grid[ni][nj] == need:
                    visited[ni][nj][nm] = True
                    dq.append((ni, nj, nm))

    # If BFS finishes without reaching (H-1,W-1)
    print("No")

if __name__ == "__main__":
    main()