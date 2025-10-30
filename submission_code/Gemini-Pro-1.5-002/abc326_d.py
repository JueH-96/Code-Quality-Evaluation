# YOUR CODE HERE
def solve():
    n = int(input())
    r = input()
    c = input()

    grid = [['.' for _ in range(n)] for _ in range(n)]

    for i in range(n):
        grid[i][0] = r[i]
    for j in range(n):
        grid[0][j] = c[j]

    def is_valid(g):
        for i in range(n):
            counts = {'A': 0, 'B': 0, 'C': 0}
            for j in range(n):
                if g[i][j] != '.':
                    counts[g[i][j]] += 1
            if any(counts[k] != 1 for k in counts):
                return False
        
        for j in range(n):
            counts = {'A': 0, 'B': 0, 'C': 0}
            for i in range(n):
                if g[i][j] != '.':
                    counts[g[i][j]] += 1
            if any(counts[k] != 1 for k in counts):
                return False
        return True

    def find_solution(row, col):
        if row == n:
            if is_valid(grid):
                print("Yes")
                for row_str in grid:
                    print("".join(row_str))
                return True
            else:
                return False

        next_row = row
        next_col = col + 1
        if next_col == n:
            next_row += 1
            next_col = 0
        
        if grid[row][col] != '.':
            return find_solution(next_row, next_col)
        
        for char in ['A', 'B', 'C']:
            grid[row][col] = char
            if find_solution(next_row, next_col):
                return True
            grid[row][col] = '.'
        
        if find_solution(next_row, next_col):
            return True
        
        return False

    if not find_solution(1, 1):
        print("No")

solve()