N, T = map(int, input().split())
A = list(map(int, input().split()))

# Map each number to its position (row, col) - 0-indexed
num_to_pos = {}
for i in range(N):
    for j in range(N):
        num = N * i + (j + 1)
        num_to_pos[num] = (i, j)

# Track counts for rows, columns, and diagonals
row_count = [0] * N
col_count = [0] * N
main_diag_count = 0  # top-left to bottom-right
anti_diag_count = 0  # top-right to bottom-left

result = -1
for turn in range(T):
    num = A[turn]
    row, col = num_to_pos[num]
    
    # Update counts
    row_count[row] += 1
    col_count[col] += 1
    
    # Check if on main diagonal
    if row == col:
        main_diag_count += 1
    
    # Check if on anti-diagonal
    if row + col == N - 1:
        anti_diag_count += 1
    
    # Check for Bingo
    if (row_count[row] == N or 
        col_count[col] == N or 
        main_diag_count == N or 
        anti_diag_count == N):
        result = turn + 1
        break

print(result)