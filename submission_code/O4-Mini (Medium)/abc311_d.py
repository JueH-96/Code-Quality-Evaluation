from collections import deque
import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    grid = [input().rstrip() for _ in range(N)]
    # visited_stop[r][c]: we've enqueued this resting position
    visited_stop = [[False]*M for _ in range(N)]
    # touched[r][c]: this ice cell was ever passed through or rested on
    touched = [[False]*M for _ in range(N)]
    # Directions: up, down, left, right
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    dq = deque()
    # Starting at (2,2) in 1-based => (1,1) in 0-based
    sr, sc = 1, 1
    visited_stop[sr][sc] = True
    touched[sr][sc] = True
    dq.append((sr, sc))
    while dq:
        r, c = dq.popleft()
        # Try each slide direction
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            # if immediately rock, no move
            if grid[nr][nc] == '#':
                continue
            # slide until next is rock
            while grid[nr][nc] == '.':
                # mark this cell touched
                if not touched[nr][nc]:
                    touched[nr][nc] = True
                nr += dr
                nc += dc
            # now (nr,nc) is rock; step back to the last ice
            stop_r, stop_c = nr - dr, nc - dc
            # enqueue this new resting pos if unseen
            if not visited_stop[stop_r][stop_c]:
                visited_stop[stop_r][stop_c] = True
                dq.append((stop_r, stop_c))
    # count touched ice cells
    ans = 0
    for i in range(N):
        for j in range(M):
            if touched[i][j]:
                ans += 1
    print(ans)

if __name__ == "__main__":
    main()