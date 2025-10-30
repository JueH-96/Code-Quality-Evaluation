from collections import deque
import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    grid = [list(input().rstrip('
')) for _ in range(N)]
    
    # Directions: up, down, left, right
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # visited[i][j]: whether we've enqueued (i,j) as a resting position
    visited = [[False]*M for _ in range(N)]
    # touched[i][j]: whether the ice square (i,j) has been touched (passed or rested on)
    touched = [[False]*M for _ in range(N)]
    
    # BFS queue for resting positions
    q = deque()
    # start at (2,2) in 1-based, so (1,1) in 0-based
    start_i, start_j = 1, 1
    visited[start_i][start_j] = True
    touched[start_i][start_j] = True
    q.append((start_i, start_j))
    
    while q:
        i, j = q.popleft()
        # Try all four slide moves
        for dx, dy in dirs:
            x, y = i, j
            # slide along the direction until the next cell is rock
            while True:
                nx, ny = x + dx, y + dy
                if grid[nx][ny] == '#':
                    # we hit rock; stop before it
                    break
                # we can move onto (nx, ny)
                x, y = nx, ny
                # mark the square as touched
                if not touched[x][y]:
                    touched[x][y] = True
            # (x,y) is now the resting position after this slide
            if not visited[x][y]:
                visited[x][y] = True
                q.append((x, y))
    
    # count all touched ice squares
    ans = sum(1 for i in range(N) for j in range(M) if touched[i][j] and grid[i][j] == '.')
    print(ans)

if __name__ == "__main__":
    main()