def solve(N, R, C):
    def is_valid(grid):
        for i in range(N):
            row = [grid[i][j] for j in range(N) if grid[i][j] != '.']
            col = [grid[j][i] for j in range(N) if grid[j][i] != '.']
            if len(set(row)) != len(row) or len(set(col)) != len(col):
                return False
        return True

    def backtrack(grid, row):
        if row == N:
            if is_valid(grid):
                return grid
            return None
        for perm in permutations('ABC'):
            new_grid = [row[:] for row in grid]
            for i, char in enumerate(perm):
                if i == 0 and char != R[row]:
                    break
                if i == 0 and char != C[row]:
                    break
                if char == R[row] and i != 0:
                    break
                if char == C[row] and i != 0:
                    break
                new_grid[row][i] = char
            result = backtrack(new_grid, row + 1)
            if result:
                return result
        return None

    from itertools import permutations
    grid = [['.' for _ in range(N)] for _ in range(N)]
    result = backtrack(grid, 0)
    if result:
        print("Yes")
        for row in result:
            print(''.join(row))
    else:
        print("No")

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    R = data[index]
    index += 1
    C = data[index]
    solve(N, R, C)