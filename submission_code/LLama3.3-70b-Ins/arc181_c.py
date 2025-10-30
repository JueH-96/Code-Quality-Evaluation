import sys

def solve(N, P, Q):
    # Create a grid filled with zeros
    grid = [[0] * N for _ in range(N)]

    # Iterate over the rows in the order of P
    for i, p in enumerate(P):
        # Iterate over the columns in the order of Q
        for j, q in enumerate(Q):
            # If this is the first row or column, set the cell to 1
            if i == 0 or j == 0:
                grid[p-1][q-1] = 1
            # If the cell above is 1, set this cell to 1
            elif grid[p-2][q-1] == 1:
                grid[p-1][q-1] = 1
            # If the cell to the left is 1, set this cell to 1
            elif grid[p-1][q-2] == 1:
                grid[p-1][q-1] = 1

    # Convert the grid to a string
    result = '
'.join(''.join(map(str, row)) for row in grid)

    return result

def main():
    # Read the input
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    # Solve the problem
    result = solve(N, P, Q)

    # Print the result
    print(result)

if __name__ == '__main__':
    main()