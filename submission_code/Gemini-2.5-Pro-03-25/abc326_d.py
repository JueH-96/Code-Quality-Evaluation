# YOUR CODE HERE

import sys
import itertools

# Function implementing the logic
def solve():
    # Read input N, R, C from standard input
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Generate all permutations of indices 0 to N-1. 
    # Each permutation represents possible column placements for a character across N rows.
    perms = list(itertools.permutations(range(N)))

    # Iterate through all combinations of 3 permutations (one for each character A, B, C)
    # pA, pB, pC are tuples representing permutations. 
    # pA[i] is the column index where 'A' is placed in row i.
    # Similarly for pB and pC.
    for pA in perms: 
        for pB in perms: 
            
            # Optimization: Check early if pA and pB clash in any row.
            # This check enforces part of Condition 1: For each row i, the cells for A, B, C must be in distinct columns.
            # Here we check if A and B are placed in the same column in any row.
            valid_AB = True
            for i in range(N):
                if pA[i] == pB[i]:
                    valid_AB = False
                    break # Found a row i where A and B clash.
            if not valid_AB:
                continue # This combination of pA and pB is invalid, try the next pB.

            # If pA and pB are valid so far, proceed to check with pC.
            for pC in perms: 
                
                # Check the remaining parts of Condition 1: 
                # Ensure A, B, C are in distinct columns for every row i.
                valid_ABC = True
                for i in range(N):
                    # We already know pA[i] != pB[i] from the check above.
                    # Now check if pC[i] clashes with pA[i] or pB[i].
                    if pA[i] == pC[i] or pB[i] == pC[i]: 
                        valid_ABC = False
                        break # Found a row i where C clashes with A or B.
                if not valid_ABC:
                    continue # This pC is invalid with the current pA, pB. Try the next pC.

                # At this point, Condition 1 holds for the triplet (pA, pB, pC).
                # Now check Condition 2: The leftmost character in row i must match R[i].
                valid_R = True
                for i in range(N):
                    # Find the minimum column index among the three characters placed in row i
                    min_col = min(pA[i], pB[i], pC[i])
                    
                    # Determine which character ('A', 'B', or 'C') is at this minimum column index (leftmost position)
                    leftmost_char = ''
                    if pA[i] == min_col:
                        leftmost_char = 'A'
                    elif pB[i] == min_col:
                        leftmost_char = 'B'
                    else: # pC[i] must be the minimum column index
                        leftmost_char = 'C'

                    # Check if this leftmost character matches the required character R[i]
                    if leftmost_char != R[i]:
                        valid_R = False
                        break # Mismatch found for row i, this triplet (pA, pB, pC) is invalid.
                
                if not valid_R:
                    continue # Try the next pC.

                # At this point, Conditions 1 and 2 hold.
                # Now check Condition 3: The topmost character in column j must match C[j].
                
                # To efficiently check Condition 3, we compute inverse permutations.
                # inv_pX[j] gives the row index i such that pX[i] = j.
                # This tells us which row contains character X in column j.
                inv_pA = [0] * N
                inv_pB = [0] * N
                inv_pC = [0] * N
                for i in range(N):
                    inv_pA[pA[i]] = i # Store row index i for column pA[i] where 'A' is placed
                    inv_pB[pB[i]] = i # Store row index i for column pB[i] where 'B' is placed
                    inv_pC[pC[i]] = i # Store row index i for column pC[i] where 'C' is placed

                valid_C = True
                for j in range(N): # Iterate through columns
                    # Find the minimum row index among the three characters placed in column j
                    min_row = min(inv_pA[j], inv_pB[j], inv_pC[j])
                    
                    # Determine which character ('A', 'B', or 'C') is at this minimum row index (topmost position)
                    topmost_char = ''
                    if inv_pA[j] == min_row:
                        topmost_char = 'A'
                    elif inv_pB[j] == min_row:
                        topmost_char = 'B'
                    else: # inv_pC[j] must be the minimum row index
                        topmost_char = 'C'
                        
                    # Check if this topmost character matches the required character C[j]
                    if topmost_char != C[j]:
                        valid_C = False
                        break # Mismatch found for column j, this triplet (pA, pB, pC) is invalid.
                
                if not valid_C:
                    continue # Try the next pC.

                # If we have reached here, all three conditions are satisfied for the triplet (pA, pB, pC).
                # We found a valid grid configuration.
                print("Yes")
                
                # Construct the grid based on the found permutations pA, pB, pC
                grid = [['.' for _ in range(N)] for _ in range(N)]
                for i in range(N):
                    grid[i][pA[i]] = 'A'
                    grid[i][pB[i]] = 'B'
                    grid[i][pC[i]] = 'C'
                
                # Print the resulting grid row by row
                for i in range(N):
                    print("".join(grid[i]))
                
                # Exit the function successfully as we found one solution as required.
                return 

    # If the loops complete without finding and returning a solution, it means no such grid exists.
    print("No")

# Call the function to run the solution based on stdin
solve()