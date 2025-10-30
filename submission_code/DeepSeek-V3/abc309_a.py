# YOUR CODE HERE
A, B = map(int, input().split())
# Define the positions of the numbers on the 3x3 grid
positions = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (2, 0),
    8: (2, 1),
    9: (2, 2)
}
# Get the positions of A and B
row_A, col_A = positions[A]
row_B, col_B = positions[B]
# Check if they are in the same row and adjacent columns
if row_A == row_B and abs(col_A - col_B) == 1:
    print("Yes")
else:
    print("No")