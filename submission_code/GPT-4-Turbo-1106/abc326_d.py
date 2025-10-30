from itertools import permutations

def solve(N, R, C):
    # Generate all possible permutations of 'ABC' of length N
    perms = list(permutations('ABC' * (N // 3)))
    
    # Check each permutation to see if it satisfies the conditions
    for perm in perms:
        grid = [['.' for _ in range(N)] for _ in range(N)]
        valid = True
        
        # Place the characters in the grid according to the permutation
        for i in range(N):
            grid[i][i] = perm[i]
            grid[i][(i + 1) % N] = perm[(i + 1) % N]
            grid[i][(i + 2) % N] = perm[(i + 2) % N]
        
        # Check if the grid satisfies the row condition
        for i in range(N):
            if R[i] != '.' and grid[i][0] != R[i]:
                valid = False
                break
        
        # Check if the grid satisfies the column condition
        if valid:
            for i in range(N):
                if C[i] != '.' and grid[0][i] != C[i]:
                    valid = False
                    break
        
        # If the grid is valid, print it and return
        if valid:
            print("Yes")
            for row in grid:
                print(''.join(row))
            return
    
    # If no valid grid is found, print No
    print("No")

# Read input
N = int(input().strip())
R = input().strip()
C = input().strip()

# Solve the problem
solve(N, R, C)