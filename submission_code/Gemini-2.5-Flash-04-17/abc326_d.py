import itertools
import sys

# Read input
N = int(sys.stdin.readline())
R = sys.stdin.readline().strip()
C = sys.stdin.readline().strip()

# Generate all permutations of (0, 1, ..., N-1)
perms = list(itertools.permutations(range(N)))

# Iterate through all possible permutations for A, B, and C
for pA_tuple in perms:
    pA = list(pA_tuple)

    # Optimization: Construct pA_inv once per pA
    pA_inv = [0] * N
    for i in range(N):
        pA_inv[pA[i]] = i

    for pB_tuple in perms:
        pB = list(pB_tuple)

        # Optimization: Construct pB_inv once per pA, pB pair
        pB_inv = [0] * N
        for i in range(N):
            pB_inv[pB[i]] = i

        for pC_tuple in perms:
            pC = list(pC_tuple)

            # Check 5a: distinct column indices in each row
            distinct_cols_ok = True
            for i in range(N):
                if pA[i] == pB[i] or pA[i] == pC[i] or pB[i] == pC[i]:
                    distinct_cols_ok = False
                    break
            if not distinct_cols_ok:
                continue # Try next pC

            # Check 5b: row leftmost constraint
            row_leftmost_ok = True
            for i in range(N):
                char_cols = []
                if R[i] == 'A':
                    char_cols = [pA[i], pB[i], pC[i]]
                elif R[i] == 'B':
                    char_cols = [pB[i], pA[i], pC[i]]
                else: # R[i] == 'C'
                    char_cols = [pC[i], pA[i], pB[i]]

                # The first element in char_cols must be the minimum of the three
                # This checks if the character R[i] is at the minimum column index
                if not (char_cols[0] < char_cols[1] and char_cols[0] < char_cols[2]):
                    row_leftmost_ok = False
                    break

            if not row_leftmost_ok:
                continue # Try next pC

            # Construct pC_inv (5c)
            pC_inv = [0] * N
            for i in range(N):
                pC_inv[pC[i]] = i

            # Check 5d: column topmost constraint
            col_topmost_ok = True
            for j in range(N):
                char_rows = []
                if C[j] == 'A':
                    char_rows = [pA_inv[j], pB_inv[j], pC_inv[j]]
                elif C[j] == 'B':
                    char_rows = [pB_inv[j], pA_inv[j], pC_inv[j]]
                else: # C[j] == 'C'
                    char_rows = [pC_inv[j], pA_inv[j], pB_inv[j]]

                # The first element in char_rows must be the minimum of the three
                # This checks if the character C[j] is at the minimum row index
                if not (char_rows[0] < char_rows[1] and char_rows[0] < char_rows[2]):
                    col_topmost_ok = False
                    break

            if not col_topmost_ok:
                continue # Try next pC

            # If all checks pass (5e)
            print("Yes")
            grid = [['.' for _ in range(N)] for _ in range(N)]
            for i in range(N):
                grid[i][pA[i]] = 'A'
                grid[i][pB[i]] = 'B'
                grid[i][pC[i]] = 'C'
            for row in grid:
                print("".join(row))

            # Exit the program after finding and printing one solution
            sys.exit()

# If loops complete without finding a solution
print("No")