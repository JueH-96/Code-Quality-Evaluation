def solve():
    n = int(input())
    size = 3**n
    grid = [['.' for _ in range(size)] for _ in range(size)]

    def fill_carpet(row_start, col_start, size, level):
        if level == 0:
            grid[row_start][col_start] = '#'
            return

        sub_size = size // 3
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                fill_carpet(row_start + i * sub_size, col_start + j * sub_size, sub_size, level - 1)

    fill_carpet(0, 0, size, n)

    for row in grid:
        print("".join(row))

solve()