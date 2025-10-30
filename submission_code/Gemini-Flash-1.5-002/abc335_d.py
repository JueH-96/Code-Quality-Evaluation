# YOUR CODE HERE
n = int(input())
grid = [[0 for _ in range(n)] for _ in range(n)]
center_row = (n + 1) // 2 - 1
center_col = (n + 1) // 2 - 1
grid[center_row][center_col] = 'T'
curr = 1
row, col = center_row, center_col - 1
while curr < n * n:
    while col >= 0 and grid[row][col] == 0:
        grid[row][col] = curr
        curr += 1
        col -= 1
    row += 1
    col += 1
    while row < n and grid[row][col] == 0:
        grid[row][col] = curr
        curr += 1
        row += 1
    col += 1
    row -= 1
    while col < n and grid[row][col] == 0:
        grid[row][col] = curr
        curr += 1
        col += 1
    row -= 1
    col -= 1
    while row >= 0 and grid[row][col] == 0:
        grid[row][col] = curr
        curr += 1
        row -= 1
    col -= 1
    row += 1
for row in grid:
    print(*row)