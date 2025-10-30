# YOUR CODE HERE
def solve():
    grid = []
    for _ in range(9):
        grid.append(list(map(int, input().split())))

    def check_row(row):
        return set(grid[row]) == set(range(1, 10))

    def check_col(col):
        return set(grid[row][col] for row in range(9)) == set(range(1, 10))

    def check_subgrid(row_start, col_start):
        seen = set()
        for i in range(row_start, row_start + 3):
            for j in range(col_start, col_start + 3):
                seen.add(grid[i][j])
        return seen == set(range(1, 10))

    for i in range(9):
        if not check_row(i):
            print("No")
            return

    for j in range(9):
        if not check_col(j):
            print("No")
            return

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if not check_subgrid(i, j):
                print("No")
                return

    print("Yes")

solve()