N, T = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the grid and counters for rows, columns, and diagonals
grid = [[0 for _ in range(N)] for _ in range(N)]
row_counter = [0 for _ in range(N)]
col_counter = [0 for _ in range(N)]
diag_counter = [0, 0]  # for main and secondary diagonals

# Function to check if Bingo is achieved
def check_bingo():
    for i in range(N):
        if row_counter[i] == N or col_counter[i] == N:
            return True
    if diag_counter[0] == N or diag_counter[1] == N:
        return True
    return False

# Process each turn
for turn in range(T):
    num = A[turn]
    row = (num - 1) // N
    col = (num - 1) % N

    # Mark the cell and update counters
    if grid[row][col] == 0:
        grid[row][col] = 1
        row_counter[row] += 1
        col_counter[col] += 1
        if row == col:
            diag_counter[0] += 1
        if row + col == N - 1:
            diag_counter[1] += 1

    # Check if Bingo is achieved
    if check_bingo():
        print(turn + 1)
        exit()

# Bingo not achieved within T turns
print(-1)