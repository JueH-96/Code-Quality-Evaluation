# YOUR CODE HERE
def solve(N, R, C):
    def is_valid(grid):
        for i in range(N):
            row = ''.join(grid[i])
            col = ''.join(grid[j][i] for j in range(N))
            if row.count('A') != 1 or row.count('B') != 1 or row.count('C') != 1:
                return False
            if col.count('A') != 1 or col.count('B') != 1 or col.count('C') != 1:
                return False
        return True

    def place_char(grid, i, j, char):
        grid[i][j] = char
        if is_valid(grid):
            return True
        grid[i][j] = '.'
        return False

    def backtrack(grid, i, j):
        if i == N:
            return True
        if j == N:
            return backtrack(grid, i + 1, 0)

        if grid[i][j] != '.':
            return backtrack(grid, i, j + 1)

        for char in 'ABC':
            if place_char(grid, i, j, char):
                if backtrack(grid, i, j + 1):
                    return True
                grid[i][j] = '.'

        return False

    grid = [['.' for _ in range(N)] for _ in range(N)]

    for i in range(N):
        grid[i][0] = R[i]
        grid[0][i] = C[i]

    if not is_valid(grid):
        print("No")
        return

    if backtrack(grid, 0, 1):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

# Read input
N = int(input())
R = input()
C = input()

# Solve the problem
solve(N, R, C)