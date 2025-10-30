# Read the input values for A and B
A, B = map(int, input().split())

# Define the 3x3 board
board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Initialize a flag to check if A and B are adjacent horizontally
adjacent = False

# Check if A and B are adjacent horizontally
for row in board:
    for i in range(2):  # Only need to check the first two elements in each row
        if row[i] == A and row[i + 1] == B:
            adjacent = True
            break

# Output the result
print("Yes" if adjacent else "No")