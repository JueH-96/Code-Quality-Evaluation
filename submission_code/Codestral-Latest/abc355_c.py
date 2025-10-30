import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
T = int(data[index])
index += 1

A = list(map(int, data[index:index + T]))

# Convert the 1D list A to a 2D grid
grid = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        grid[i][j] = N * i + j + 1

# Create sets to track marked rows, columns, and diagonals
marked_rows = [0] * N
marked_cols = [0] * N
marked_diagonal1 = 0
marked_diagonal2 = 0

# Function to check if Bingo is achieved
def check_bingo(row, col):
    global marked_diagonal1, marked_diagonal2
    marked_rows[row] += 1
    marked_cols[col] += 1
    if row == col:
        marked_diagonal1 += 1
    if row == N - col - 1:
        marked_diagonal2 += 1
    if marked_rows[row] == N or marked_cols[col] == N or marked_diagonal1 == N or marked_diagonal2 == N:
        return True
    return False

# Process each turn
for turn in range(T):
    value = A[turn]
    for i in range(N):
        for j in range(N):
            if grid[i][j] == value:
                if check_bingo(i, j):
                    print(turn + 1)
                    sys.exit()

print(-1)