import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
R_str = data[index]
index += 1
C_str = data[index]

# Map symbols to indices
sym_to_idx = {'A': 0, 'B': 1, 'C': 2}
R_idx = [sym_to_idx[c] for c in R_str]
C_idx = [sym_to_idx[c] for c in C_str]

# Initialize pi: pi[s][i] is the column for symbol s in row i
pi = [[-1 for _ in range(N)] for _ in range(3)]

# Initial available columns for each symbol, using bitmasks
full_mask = (1 << N) - 1
available_init = (full_mask, full_mask, full_mask)

def check_columns(pi):
    # Compute inverse permutations: inv_pi[s][j] is the row where symbol s is in column j
    inv_pi_list = []
    for s in range(3):
        inv_pi_s = [-1] * N
        for i in range(N):
            col = pi[s][i]
            inv_pi_s[col] = i
        inv_pi_list.append(inv_pi_s)
    
    # Check for each column j
    for j in range(N):
        # Find the symbol with the minimum row index in column j
        s_min = min(range(3), key=lambda s: inv_pi_list[s][j])
        if s_min != C_idx[j]:
            return False
    return True

def backtrack(row, pi, available):
    if row == N:
        if check_columns(pi):
            return True
        else:
            return False
    
    S_min = R_idx[row]
    other_syms = [s for s in range(3) if s != S_min]
    sym1, sym2 = other_syms[0], other_syms[1]
    
    for c_min in range(N):
        if (available[S_min] & (1 << c_min)) == 0:  # Column not available for S_min
            continue
        for c1 in range(c_min + 1, N):  # c1 > c_min
            if (available[sym1] & (1 << c1)) == 0:  # Column not available for sym1
                continue
            for c2 in range(c_min + 1, N):  # c2 > c_min
                if c2 == c1:  # Columns must be distinct
                    continue
                if (available[sym2] & (1 << c2)) == 0:  # Column not available for sym2
                    continue
                # Assign columns to symbols for this row
                pi[S_min][row] = c_min
                pi[sym1][row] = c1
                pi[sym2][row] = c2
                # Create new available with assigned columns removed
                new_available = tuple(
                    available[s] & ~(1 << pi[s][row]) for s in range(3)
                )
                # Recurse to next row
                if backtrack(row + 1, pi, new_available):
                    return True
                # Backtrack: unset assignments
                pi[S_min][row] = -1
                pi[sym1][row] = -1
                pi[sym2][row] = -1
    
    # No valid assignment found for this row
    return False

# Call backtrack starting from row 0
if backtrack(0, pi, available_init):
    # Output the grid
    print("Yes")
    for i in range(N):  # For each row
        grid_row = ['.'] * N
        for s in range(3):  # For each symbol
            col = pi[s][i]
            if col != -1:  # Set the character for the symbol
                grid_row[col] = chr(ord('A') + s)
        print(''.join(grid_row))
else:
    print("No")