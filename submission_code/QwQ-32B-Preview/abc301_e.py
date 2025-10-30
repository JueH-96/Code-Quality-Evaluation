import sys
from collections import deque

def main():
    import sys
    from io import StringIO
    import copy
    sys.setrecursionlimit(1 << 25)
    # sys.stdin = StringIO(testcase)
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]

    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Find positions of S, G, and o's
    important_points = []
    o_indices = {}
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                important_points.append((i, j))
                S = len(important_points) - 1
            elif grid[i][j] == 'G':
                important_points.append((i, j))
                G = len(important_points) - 1
            elif grid[i][j] == 'o':
                important_points.append((i, j))
                o_indices[len(important_points) - 1] = len(o_indices)

    N = len(o_indices)  # Number of o's
    M = len(important_points)  # Total important points

    # Function to perform BFS from a start position to all other important points
    def bfs(start):
        dist = [float('inf')] * M
        dist[start] = 0
        queue = deque([start])
        visited = [False] * M
        visited[start] = True
        while queue:
            current = queue.popleft()
            x, y = important_points[current]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                    nxt = important_points.index((nx, ny)) if (nx, ny) in important_points else -1
                    if nxt != -1 and not visited[nxt]:
                        dist[nxt] = dist[current] + 1
                        visited[nxt] = True
                        queue.append(nxt)
        return dist

    # Compute all-pairs shortest paths
    dist = []
    for i in range(M):
        dist.append(bfs(i))

    # DP[mask][k]: min moves to reach k with mask of o's visited
    INF = float('inf')
    dp = [[INF] * M for _ in range(1 << N)]
    dp[0][S] = 0  # Start at S with no o's visited

    # Iterate over all masks
    for mask in range(1 << N):
        for m in range(M):
            if dp[mask][m] == INF:
                continue
            for k in range(M):
                if grid[important_points[k][0]][important_points[k][1]] == 'o':
                    o_index = o_indices[k]
                    new_mask = mask | (1 << o_index)
                else:
                    new_mask = mask
                if dp[mask][m] + dist[m][k] < dp[new_mask][k]:
                    dp[new_mask][k] = dp[mask][m] + dist[m][k]

    # Find the maximum number of o's visited with dp[mask][G] <= T
    max_o = -1
    for mask in range(1 << N):
        if dp[mask][G] <= T:
            count = bin(mask).count('1')
            if count > max_o:
                max_o = count

    print(max_o if max_o != -1 else -1)

if __name__ == '__main__':
    main()