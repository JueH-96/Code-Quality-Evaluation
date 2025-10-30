def solve():
    h, w, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    grid = [[1] * w for _ in range(h)]
    
    def destroy_wall(r, c):
        if 0 <= r < h and 0 <= c < w:
            grid[r][c] = 0

    for r, c in queries:
        r -= 1
        c -= 1
        
        if grid[r][c] == 1:
            destroy_wall(r, c)
        else:
            # Up
            for i in range(r - 1, -1, -1):
                if grid[i][c] == 1:
                    destroy_wall(i, c)
                    break
            # Down
            for i in range(r + 1, h):
                if grid[i][c] == 1:
                    destroy_wall(i, c)
                    break
            # Left
            for j in range(c - 1, -1, -1):
                if grid[r][j] == 1:
                    destroy_wall(r, j)
                    break
            # Right
            for j in range(c + 1, w):
                if grid[r][j] == 1:
                    destroy_wall(r, j)
                    break

    count = 0
    for row in grid:
        count += sum(row)
    print(count)

solve()