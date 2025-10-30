import sys

def solve():
    """
    Reads a grid, shifts the outer layer clockwise, and prints the result.
    """
    try:
        # Read the size of the grid, N.
        N = int(sys.stdin.readline())
        if N == 0: return # Handle empty input case

        # Read the N x N grid. Each row is read as a string and converted to a list of characters.
        A = [list(sys.stdin.readline().strip()) for _ in range(N)]
    except (IOError, ValueError):
        # This is a fallback for environments where stdin might not behave as expected,
        # or for empty input, though the problem constraints (N>=2) make this unlikely.
        return

    # Create a new grid B as a copy of the original grid A.
    # The shifting operation will read from A and write to B to avoid
    # overwriting values prematurely.
    B = [row[:] for row in A]

    # Get the last index, which is N-1.
    n1 = N - 1

    # The following four loops perform the clockwise shift on the outer squares.

    # 1. Top row: Values move one step to the right.
    # The value at A[0][j] goes to B[0][j+1].
    for j in range(n1):
        B[0][j + 1] = A[0][j]

    # 2. Right column: Values move one step down.
    # The value at A[i][n1] goes to B[i+1][n1].
    for i in range(n1):
        B[i + 1][n1] = A[i][n1]

    # 3. Bottom row: Values move one step to the left.
    # The value at A[n1][j] goes to B[n1][j-1].
    for j in range(n1, 0, -1):
        B[n1][j - 1] = A[n1][j]

    # 4. Left column: Values move one step up.
    # The value at A[i][0] goes to B[i-1][0].
    for i in range(n1, 0, -1):
        B[i - 1][0] = A[i][0]

    # Print the final grid B, with each row's characters joined into a string.
    for row in B:
        print("".join(row))

solve()