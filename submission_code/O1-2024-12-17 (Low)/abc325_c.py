def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    S = input_data[2:]

    # Directions for adjacency (8-connected)
    directions = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1),           (0, 1), 
                  (1, -1),  (1, 0),  (1, 1)]

    visited = [[False]*W for _ in range(H)]

    def valid(i, j):
        return (0 <= i < H) and (0 <= j < W)

    def dfs(si, sj):
        stack = [(si, sj)]
        visited[si][sj] = True
        while stack:
            i, j = stack.pop()
            for di, dj in directions:
                ni, nj = i+di, j+dj
                if valid(ni, nj) and not visited[ni][nj] and S[ni][nj] == '#':
                    visited[ni][nj] = True
                    stack.append((ni, nj))

    sensor_count = 0
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#' and not visited[i][j]:
                dfs(i, j)
                sensor_count += 1

    print(sensor_count)

# Do not forget to call main()
if __name__ == "__main__":
    main()