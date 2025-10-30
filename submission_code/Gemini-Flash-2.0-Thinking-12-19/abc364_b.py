def solve():
    h, w = map(int, input().split())
    si, sj = map(int, input().split())
    grid = [input() for _ in range(h)]
    x = input()

    row = si - 1
    col = sj - 1

    for move in x:
        if move == 'L':
            if col > 0 and grid[row][col - 1] == '.':
                col -= 1
        elif move == 'R':
            if col < w - 1 and grid[row][col + 1] == '.':
                col += 1
        elif move == 'U':
            if row > 0 and grid[row - 1][col] == '.':
                row -= 1
        elif move == 'D':
            if row < h - 1 and grid[row + 1][col] == '.':
                row += 1

    print(row + 1, col + 1)

if __name__ == "__main__":
    solve()