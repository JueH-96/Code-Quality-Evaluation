import sys
input = sys.stdin.read

def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = [list(data[i]) for i in range(2, H + 2)]

    def mark_rows(grid):
        marked = set()
        for i in range(H):
            if all(grid[i][j] == grid[i][0] for j in range(W) if grid[i][j] != '.'):
                for j in range(W):
                    if grid[i][j] != '.':
                        marked.add((i, j))
        return marked

    def mark_cols(grid):
        marked = set()
        for j in range(W):
            if all(grid[i][j] == grid[0][j] for i in range(H) if grid[i][j] != '.'):
                for i in range(H):
                    if grid[i][j] != '.':
                        marked.add((i, j))
        return marked

    while True:
        marked_rows = mark_rows(grid)
        marked_cols = mark_cols(grid)

        if not marked_rows and not marked_cols:
            break

        for i, j in marked_rows | marked_cols:
            grid[i][j] = '.'

    remaining_cookies = sum(1 for i in range(H) for j in range(W) if grid[i][j] != '.')
    print(remaining_cookies)

if __name__ == "__main__":
    main()