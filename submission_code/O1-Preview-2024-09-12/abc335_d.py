# YOUR CODE HERE
N = int(input())
grid = [[0]*N for _ in range(N)]
center_row = N // 2
center_col = N // 2

start_row = 0
end_row = N - 1
start_col = 0
end_col = N - 1
num = 1

while start_row <= end_row and start_col <= end_col:
    # Traverse from left to right along the top row
    for col in range(start_col, end_col + 1):
        if start_row == center_row and col == center_col:
            continue
        grid[start_row][col] = num
        num += 1
    start_row +=1
    
    # Traverse downwards along the rightmost column
    for row in range(start_row, end_row +1):
        if row == center_row and end_col == center_col:
            continue
        grid[row][end_col] = num
        num +=1
    end_col -=1
    
    # Traverse from right to left along the bottom row
    if start_row <= end_row:
        for col in range(end_col, start_col -1, -1):
            if end_row == center_row and col == center_col:
                continue
            grid[end_row][col] = num
            num +=1
        end_row -=1
    
    # Traverse upwards along the leftmost column
    if start_col <= end_col:
        for row in range(end_row, start_row -1, -1):
            if row == center_row and start_col == center_col:
                continue
            grid[row][start_col] = num
            num +=1
        start_col +=1

for i in range(N):
    row_output = []
    for j in range(N):
        if i == center_row and j == center_col:
            row_output.append("T")
        else:
            row_output.append(str(grid[i][j]))
    print(' '.join(row_output))