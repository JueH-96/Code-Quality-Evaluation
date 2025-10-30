def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                print(i + 1, j + 1)
                return

if __name__ == '__main__':
    solve()