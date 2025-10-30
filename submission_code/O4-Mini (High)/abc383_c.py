import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W, D = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]

    # dist[i][j] will store the minimum number of moves from any H to (i,j), or -1 if unreachable.
    dist = [[-1] * W for _ in range(H)]
    q = deque()

    # Initialize BFS queue with all humidifier positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))

    # Multi-source BFS up to distance D
    while q:
        i, j = q.popleft()
        if dist[i][j] == D:
            continue
        nd = dist[i][j] + 1
        # explore 4 neighbors
        if i > 0 and grid[i-1][j] != '#' and dist[i-1][j] == -1:
            dist[i-1][j] = nd
            q.append((i-1, j))
        if i+1 < H and grid[i+1][j] != '#' and dist[i+1][j] == -1:
            dist[i+1][j] = nd
            q.append((i+1, j))
        if j > 0 and grid[i][j-1] != '#' and dist[i][j-1] == -1:
            dist[i][j-1] = nd
            q.append((i, j-1))
        if j+1 < W and grid[i][j+1] != '#' and dist[i][j+1] == -1:
            dist[i][j+1] = nd
            q.append((i, j+1))

    # Count humidified floor cells (including those with 'H')
    ans = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1:
                ans += 1

    print(ans)

if __name__ == "__main__":
    main()