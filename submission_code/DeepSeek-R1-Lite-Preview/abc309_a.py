# Define the grid
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Function to find the position of a number
def find_position(number):
    for row_index, row in enumerate(grid):
        if number in row:
            return (row_index, row.index(number))
    return (-1, -1)  # Should not happen as numbers are from 1 to 9

# Read input
A, B = map(int, input().split())

# Find positions of A and B
posA = find_position(A)
posB = find_position(B)

# Check if they are in the same row and adjacent columns
if posA[0] == posB[0] and abs(posA[1] - posB[1]) == 1:
    print("Yes")
else:
    print("No")