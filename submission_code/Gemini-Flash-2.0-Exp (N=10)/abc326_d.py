def solve():
    n = int(input())
    r = input()
    c = input()

    def check(grid):
        for row in grid:
            if row.count('A') != 1 or row.count('B') != 1 or row.count('C') != 1:
                return False
        for col in range(n):
            column = [grid[row][col] for row in range(n)]
            if column.count('A') != 1 or column.count('B') != 1 or column.count('C') != 1:
                return False
        for i in range(n):
            first_char = ""
            for j in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char != r[i]:
                return False
        for j in range(n):
            first_char = ""
            for i in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char != c[j]:
                return False
        return True

    import itertools

    chars = ['A', 'B', 'C', '.']
    for grid_tuple in itertools.product(chars, repeat=n*n):
        grid = [list(grid_tuple[i*n:(i+1)*n]) for i in range(n)]
        if check(grid):
            print("Yes")
            for row in grid:
                print("".join(row))
            return
    print("No")

solve()