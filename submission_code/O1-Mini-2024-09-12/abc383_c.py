# YOUR CODE HERE
import sys
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    H, W, D = map(int, data[:3])
    grid = data[3:]
    assert len(grid) == H
    # Find all humidifier positions
    humidifiers = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                humidifiers.append((i, j))
    if not humidifiers:
        print(0)
        return
    # Initialize distance grid
    visited = [[-1]*W for _ in range(H)]
    q = deque()
    for i,j in humidifiers:
        visited[i][j] = 0
        q.append((i, j))
    # BFS
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        i,j = q.popleft()
        current_dist = visited[i][j]
        if current_dist == D:
            continue
        for di, dj in directions:
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] != '#' and visited[ni][nj]==-1:
                    visited[ni][nj] = current_dist +1
                    q.append((ni, nj))
    # Count humidified floor cells
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and visited[i][j]!=-1 and visited[i][j] <= D:
                count +=1
    print(count)

if __name__ == "__main__":
    main()