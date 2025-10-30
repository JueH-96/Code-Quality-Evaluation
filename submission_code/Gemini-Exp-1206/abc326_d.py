def solve():
    n = int(input())
    r = input()
    c = input()

    def check(grid):
        for row in grid:
            if row.count('A') != 1 or row.count('B') != 1 or row.count('C') != 1:
                return False
        for j in range(n):
            col = [grid[i][j] for i in range(n)]
            if col.count('A') != 1 or col.count('B') != 1 or col.count('C') != 1:
                return False
        for i in range(n):
            leftmost = ''
            for j in range(n):
                if grid[i][j] != '.':
                    leftmost = grid[i][j]
                    break
            if leftmost != r[i]:
                return False
        for j in range(n):
            topmost = ''
            for i in range(n):
                if grid[i][j] != '.':
                    topmost = grid[i][j]
                    break
            if topmost != c[j]:
                return False
        return True

    def find_solution(grid, row):
        if row == n:
            if check(grid):
                print("Yes")
                for r in grid:
                    print("".join(r))
                return True
            else:
                return False

        for p in permutations(list(range(n))):
            new_grid = [list(row) for row in grid]
            chars = ['A', 'B', 'C']
            
            valid_row = True
            for j in range(n):
                new_grid[row][p[j]] = chars[j]

            
            leftmost_valid = False
            for j in range(n):
                if new_grid[row][j] != '.':
                    if new_grid[row][j] == r[row]:
                        leftmost_valid = True
                    break
            if not leftmost_valid:
                continue
            
            
            
            
            
            if find_solution(new_grid, row + 1):
                return True
        return False

    from itertools import permutations
    grid = [['.' for _ in range(n)] for _ in range(n)]
    if not find_solution(grid, 0):
        print("No")

solve()