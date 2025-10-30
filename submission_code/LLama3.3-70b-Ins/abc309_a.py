def are_adjacent_horizontally(A, B):
    # Define the positions of the numbers on the board
    positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }

    # Get the positions of A and B
    pos_A = positions[A]
    pos_B = positions[B]

    # Check if A and B are adjacent horizontally
    return pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1

# Read the input from stdin
A, B = map(int, input().split())

# Check if A and B are adjacent horizontally
if are_adjacent_horizontally(A, B):
    print("Yes")
else:
    print("No")