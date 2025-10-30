def solve():
    n = int(input())
    grid = [[0 for _ in range(n)] for _ in range(n)]
    center_row = (n - 1) // 2
    center_col = (n - 1) // 2
    grid[center_row][center_col] = 'T'
    part_number = 1
    for i in range(n):
        if i % 2 == 0:
            for j in range(n):
                if grid[i][j] == 0:
                    grid[i][j] = part_number
                    part_number += 1
        else:
            for j in range(n - 1, -1, -1):
                if grid[i][j] == 0:
                    grid[i][j] = part_number
                    part_number += 1
                    
    for i in range(n):
        row_str = ""
        for j in range(n):
            row_str += str(grid[i][j])
            if j < n - 1:
                row_str += " "
        print(row_str)

if __name__ == '__main__':
    solve()