# YOUR CODE HERE
import sys

# Read N and T
line1 = sys.stdin.readline().split()
N = int(line1[0])
T = int(line1[1])

# Read the list of announced numbers
# Using list comprehension with map is a concise way to read integers
A = list(map(int, sys.stdin.readline().split()))

# Initialize counters
# row_counts[i]: number of marked cells in row i (0-indexed)
row_counts = [0] * N
# col_counts[j]: number of marked cells in column j (0-indexed)
col_counts = [0] * N
# diag1_count: number of marked cells on the main diagonal (r == c)
diag1_count = 0
# diag2_count: number of marked cells on the anti-diagonal (r + c == N - 1)
diag2_count = 0

# Simulate turns
# The loop runs for each announced number, from the first turn (index 0 in A) to the T-th turn (index T-1 in A).
for turn in range(T):
    value = A[turn] # Get the number announced on the current turn (0-indexed)

    # Find the 0-indexed row (r) and column (c) for the value
    # The value V at 0-indexed (r, c) is N * r + c + 1.
    # Given V, r = (V - 1) // N and c = (V - 1) % N.
    r = (value - 1) // N
    c = (value - 1) % N

    # Mark the cell (r, c) by incrementing the corresponding row and column counts
    row_counts[r] += 1
    col_counts[c] += 1

    # Check if the marked cell is on the main diagonal (top-left to bottom-right)
    if r == c:
        diag1_count += 1

    # Check if the marked cell is on the anti-diagonal (top-right to bottom-left)
    # Note: A cell can be on both diagonals if N is odd and the cell is the center one (r=c=N//2).
    # We check both conditions independently.
    if r + c == N - 1:
        diag2_count += 1

    # After marking the cell and updating counts, check if Bingo is achieved.
    # Bingo happens if any row, column, main diagonal, or anti-diagonal is full (count equals N).
    # We only need to check the lines involving the cell just marked (r, c).

    # Check if the row r is full
    if row_counts[r] == N:
        print(turn + 1) # Bingo achieved on this turn (1-indexed)
        sys.exit() # Terminate the program immediately

    # Check if the column c is full
    if col_counts[c] == N:
        print(turn + 1) # Bingo achieved on this turn (1-indexed)
        sys.exit() # Terminate the program immediately

    # Check if the main diagonal is full (only needed if the cell (r, c) is on it)
    if r == c and diag1_count == N:
        print(turn + 1) # Bingo achieved on this turn (1-indexed)
        sys.exit() # Terminate the program immediately

    # Check if the anti-diagonal is full (only needed if the cell (r, c) is on it)
    if r + c == N - 1 and diag2_count == N:
        print(turn + 1) # Bingo achieved on this turn (1-indexed)
        sys.exit() # Terminate the program immediately

# If the loop finishes without any Bingo condition being met after T turns
print(-1)

# END OF YOUR CODE