# YOUR CODE HERE
N, T = map(int, input().split())
A = list(map(int, input().split()))

# Count of marked cells in each row, column, and diagonal
row_count = [0] * N
col_count = [0] * N
main_diag_count = 0
anti_diag_count = 0

# Process turns
for turn in range(T):
    value = A[turn]
    row = (value - 1) // N
    col = (value - 1) % N
    
    # Update counts
    row_count[row] += 1
    col_count[col] += 1
    
    if row == col:
        main_diag_count += 1
    
    if row + col == N - 1:
        anti_diag_count += 1
    
    # Check if bingo is achieved
    if row_count[row] == N or col_count[col] == N or main_diag_count == N or anti_diag_count == N:
        print(turn + 1)
        break
else:
    print(-1)