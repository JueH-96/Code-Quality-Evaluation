# YOUR CODE HERE
def bingo_turn(N, T, A):
    # Initialize the grid
    grid = [[0] * N for _ in range(N)]
    
    # Initialize row, column and diagonal counters
    row_count = [0] * N
    col_count = [0] * N
    diag1_count = 0
    diag2_count = 0
    
    for turn in range(T):
        num = A[turn]
        i = (num - 1) // N
        j = (num - 1) % N
        
        # Mark the cell
        grid[i][j] = 1
        
        # Update row and column counts
        row_count[i] += 1
        col_count[j] += 1
        
        # Check if it's on the main diagonal
        if i == j:
            diag1_count += 1
        
        # Check if it's on the anti-diagonal
        if i + j == N - 1:
            diag2_count += 1
        
        # Check for Bingo
        if row_count[i] == N or col_count[j] == N or diag1_count == N or diag2_count == N:
            return turn + 1
    
    return -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
A = list(map(int, data[2:]))

# Get the result
result = bingo_turn(N, T, A)

# Print the result
print(result)