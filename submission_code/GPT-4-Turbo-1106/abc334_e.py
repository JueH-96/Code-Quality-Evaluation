from sys import stdin, stdout

MOD = 998244353

def modinv(a, mod=MOD):
    return pow(a, mod - 2, mod)

def main():
    H, W = map(int, stdin.readline().split())
    grid = [stdin.readline().strip() for _ in range(H)]

    # Count the number of red cells and initialize the number of green connected components
    red_cells = sum(row.count('.') for row in grid)
    green_components = 0

    # Directions for adjacent cells
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    # Visited matrix to keep track of visited green cells
    visited = [[False] * W for _ in range(H)]

    # DFS function to explore green connected components
    def dfs(x, y):
        visited[x][y] = True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '#':
                dfs(nx, ny)

    # Find initial number of green connected components
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                green_components += 1
                dfs(i, j)

    # Calculate the expected value
    answer = green_components * modinv(red_cells)
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                neighbors = 0
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        neighbors += 1
                answer -= modinv(red_cells) * neighbors

    # Output the answer modulo 998244353
    stdout.write(f"{answer % MOD}
")

if __name__ == "__main__":
    main()