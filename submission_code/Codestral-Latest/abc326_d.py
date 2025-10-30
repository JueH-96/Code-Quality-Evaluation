import sys
from itertools import permutations

def solve(N, R, C):
    # Check if it's possible to place characters in the grid
    if R.count('A') != N or R.count('B') != N or R.count('C') != N:
        return "No"
    if C.count('A') != N or C.count('B') != N or C.count('C') != N:
        return "No"

    # Generate all possible permutations of 'ABC'
    perms = list(permutations('ABC'))

    # Try each permutation to see if it satisfies the conditions
    for perm in perms:
        grid = [['.' for _ in range(N)] for _ in range(N)]
        valid = True

        # Place characters in the grid according to the permutation
        for i in range(N):
            grid[i][0] = perm[i]

        # Check if the first column matches C
        for i in range(N):
            if grid[i][0] != C[i]:
                valid = False
                break

        if not valid:
            continue

        # Fill the rest of the grid
        for i in range(N):
            for j in range(1, N):
                if grid[i][j] == '.':
                    for char in 'ABC':
                        if char not in grid[i] and char not in [grid[k][j] for k in range(N)]:
                            grid[i][j] = char
                            break

        # Check if the first row matches R
        for j in range(N):
            if grid[0][j] != R[j]:
                valid = False
                break

        if valid:
            return "Yes
" + "
".join("".join(row) for row in grid)

    return "No"

# Read input
N = int(sys.stdin.readline().strip())
R = sys.stdin.readline().strip()
C = sys.stdin.readline().strip()

# Solve the problem
result = solve(N, R, C)

# Print the result
print(result)