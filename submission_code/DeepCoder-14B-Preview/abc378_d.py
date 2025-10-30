def main():
    import sys
    sys.setrecursionlimit(10000)  # Increase recursion limit if necessary

    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    total = 0

    directions = [ (-1, 0), (1, 0), (0, -1), (0, 1) ]

    def dfs(i, j, visited, steps):
        if steps == K:
            return 1
        count = 0
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and (ni, nj) not in visited:
                    new_visited = visited.copy()
                    new_visited.add((ni, nj))
                    count += dfs(ni, nj, new_visited, steps + 1)
        return count

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                initial_visited = set()
                initial_visited.add((i, j))
                total += dfs(i, j, initial_visited, 0)

    print(total)

if __name__ == '__main__':
    main()