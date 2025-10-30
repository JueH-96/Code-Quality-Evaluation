# YOUR CODE HERE
import sys

# Function containing the core logic
def solve():
    # Read problem parameters N (grid size) and M (number of constraints)
    # N can be very large (up to 10^9), M is limited (up to 2*10^5)
    N, M = map(int, sys.stdin.readline().split())
    
    # Dictionary to store the minimum number of black cells required for each row (Lower bound L_r)
    # The condition for a valid coloring implies that for each row r, the first B_r cells are black
    # and the rest are white. So B_r represents the number of black cells in row r.
    # The constraint for cell (X, Y) being Black ('B') means B_X >= Y.
    # We store the maximum required Y for each row X in min_B[X].
    # B_X >= L_X where L_X = min_B[X]
    min_B = {} 
    
    # Dictionary to store the maximum number of black cells allowed for each row (Upper bound U_r)
    # The constraint for cell (X, Y) being White ('W') means B_X < Y, which is equivalent to B_X <= Y-1.
    # We store the minimum required Y-1 for each row X in max_B[X].
    # B_X <= U_X where U_X = max_B[X]
    max_B = {} 

    # Keep track of rows that have at least one constraint applied to them.
    # We only need to focus on these rows as others have default bounds [0, N].
    rows_with_constraints_set = set()

    # Process each of the M constraints
    for _ in range(M):
        # Read constraint details: row X, column Y, color C ('B' or 'W')
        line = sys.stdin.readline().split() 
        X = int(line[0])
        Y = int(line[1])
        C = line[2]
        
        # Add the row index X to the set of constrained rows
        rows_with_constraints_set.add(X)

        # Update bounds based on the constraint
        if C == 'B':
            # If cell (X, Y) must be black, row X must have at least Y black cells.
            # Update L_X = max(current L_X, Y)
            # We use .get(X, 0) because the default lower bound for any row is 0.
            min_B[X] = max(min_B.get(X, 0), Y) 
        else: # C == 'W'
            # If cell (X, Y) must be white, row X must have at most Y-1 black cells.
            # Update U_X = min(current U_X, Y-1)
            # We use .get(X, N) because the default upper bound for any row is N.
            max_B[X] = min(max_B.get(X, N), Y - 1)

    # Check initial feasibility: For each constrained row r, must have L_r <= U_r
    # This ensures that the constraints on a single row are not contradictory.
    # For example, if (r, 5, B) and (r, 3, W) are given, then L_r=5 and U_r=2.
    # Since L_r > U_r, this check will detect the impossibility.
    for r in rows_with_constraints_set:
        L_r = min_B.get(r, 0) # Default L_r is 0 if no 'B' constraint on row r
        U_r = max_B.get(r, N) # Default U_r is N if no 'W' constraint on row r
        # If lower bound exceeds upper bound for any row, it's impossible
        if L_r > U_r:
            print("No")
            return

    # Check the monotonicity condition derived from row and column properties
    # The combined row and column conditions imply that the number of black cells per row
    # must be a non-increasing sequence: B_1 >= B_2 >= ... >= B_N.
    # This property, along with the bounds L_r <= B_r <= U_r, imposes a constraint between rows.
    # Specifically, for any two rows r1 < r2, we must have B_{r1} >= B_{r2}.
    # Since B_{r1} <= U_{r1} and B_{r2} >= L_{r2}, the tightest requirement is U_{r1} >= L_{r2}.
    # This condition is necessary and sufficient (along with L_r <= U_r) for existence of such sequence B_r.
    # We only need to check this condition for pairs of rows r1, r2 that have constraints,
    # as rows without constraints have bounds [0, N] which do not restrict other rows significantly.

    # Get a sorted list of unique rows that have constraints
    sorted_rows = sorted(list(rows_with_constraints_set))
    p = len(sorted_rows) # Number of unique constrained rows

    # If there are 0 or 1 constrained rows, the monotonicity condition involves no pairs (r1 < r2),
    # so it is trivially satisfied. We already checked L_r <= U_r individually.
    if p <= 1: 
        print("Yes")
        return

    # To efficiently check U_{r1} >= L_{r2} for all r1 < r2 where r1, r2 are constrained rows,
    # we can use suffix maximums of the lower bounds L_r.
    # Let the sorted constrained rows be k_1 < k_2 < ... < k_p.
    # The condition U_{k_i} >= L_{k_j} for all j > i is equivalent to checking
    # if U_{k_i} >= max(L_{k_{j}}) for all j > i.
    # The term max(L_{k_{j}}) for all j > i is exactly the suffix maximum starting from index i+1.
    
    # Compute suffix maximums of the lower bounds L_r for the sorted constrained rows
    suffix_max_L = [0] * p # Array to store suffix maximums
    
    # Initialize the last element of the suffix max array
    last_row_idx = p - 1
    last_row = sorted_rows[last_row_idx]
    # The suffix max starting at the last element is just its L value (default 0 if no 'B' constraint)
    suffix_max_L[last_row_idx] = min_B.get(last_row, 0) 
    
    # Compute suffix maximums iteratively from right-to-left (index p-2 down to 0)
    for i in range(p - 2, -1, -1):
        current_row = sorted_rows[i]
        L_curr = min_B.get(current_row, 0) # L value for the current row k_i
        # Suffix max at index i is the maximum of L_curr and the suffix max at index i+1
        suffix_max_L[i] = max(L_curr, suffix_max_L[i+1]) 

    # Check the monotonicity condition U_{k_i} >= suffix_max_L[i+1] for i = 0 to p-2
    for i in range(p - 1): 
        current_row = sorted_rows[i] # This is row k_i
        U_curr = max_B.get(current_row, N) # U value for the current row k_i (default N if no 'W' constraint)
        
        # Check if U_{k_i} is large enough to satisfy constraints imposed by subsequent rows k_{i+1}, ..., k_p
        # Specifically, check if U_{k_i} is greater than or equal to the maximum lower bound among rows k_{i+1} to k_p.
        if U_curr < suffix_max_L[i+1]:
            # If this condition fails, it means B_{k_i} <= U_{k_i} < suffix_max_L[i+1] = max_{j>i} L_{k_j}.
            # So B_{k_i} < L_{k_j} for some j > i. This implies B_{k_i} < B_{k_j} must hold,
            # violating the non-increasing property B_{k_i} >= B_{k_j}.
            print("No") # Condition violated, impossible configuration
            return

    # If all checks passed (initial feasibility per row and monotonicity between rows)
    print("Yes") # It's possible to color the grid satisfying all conditions

# Execute the solver function
solve()