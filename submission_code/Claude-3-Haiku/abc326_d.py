import sys

def solve():
    N = int(input())
    R = input()
    C = input()

    # Create a 2D grid to store the characters
    grid = [['.' for _ in range(N)] for _ in range(N)]

    # Check if it's possible to fill the grid
    for i in range(N):
        # Find the positions of A, B, C in the current row and column
        row_a, row_b, row_c = -1, -1, -1
        col_a, col_b, col_c = -1, -1, -1
        for j in range(N):
            if R[i] == 'A':
                row_a = j
            elif R[i] == 'B':
                row_b = j
            elif R[i] == 'C':
                row_c = j
            if C[j] == 'A':
                col_a = j
            elif C[j] == 'B':
                col_b = j
            elif C[j] == 'C':
                col_c = j

        # Check if it's possible to fill the current row and column
        if row_a == -1 or row_b == -1 or row_c == -1 or col_a == -1 or col_b == -1 or col_c == -1:
            print("No")
            return

        # Fill the grid
        grid[i][row_a] = 'A'
        grid[i][row_b] = 'B'
        grid[i][row_c] = 'C'
        grid[col_a][i] = 'A'
        grid[col_b][i] = 'B'
        grid[col_c][i] = 'C'

    # Print the solution
    print("Yes")
    for row in grid:
        print(''.join(row))

solve()