# YOUR CODE HERE
N, T = map(int, input().split())
A = list(map(int, input().split()))

# Create a mapping from number to position (row, col)
num_to_pos = {}
for i in range(N):
    for j in range(N):
        num = N * i + (j + 1)
        num_to_pos[num] = (i, j)

# Track marked cells
marked = [[False] * N for _ in range(N)]

# Track counts for rows, columns, and diagonals
row_count = [0] * N
col_count = [0] * N
diag1_count = 0  # top-left to bottom-right
diag2_count = 0  # top-right to bottom-left

for turn in range(T):
    num = A[turn]
    if num in num_to_pos:
        row, col = num_to_pos[num]
        
        # Mark the cell
        marked[row][col] = True
        
        # Update counts
        row_count[row] += 1
        col_count[col] += 1
        
        # Check if on main diagonal (top-left to bottom-right)
        if row == col:
            diag1_count += 1
            
        # Check if on anti-diagonal (top-right to bottom-left)
        if row + col == N - 1:
            diag2_count += 1
        
        # Check for bingo
        if (row_count[row] == N or 
            col_count[col] == N or 
            diag1_count == N or 
            diag2_count == N):
            print(turn + 1)
            exit()

print(-1)