# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N (number of tooth pairs) and X (maximum allowed difference between adjacent upper teeth)
    N, X = map(int, sys.stdin.readline().split())
    
    # List to store tooth pairs information (U_i, D_i)
    pairs = []
    # Variable to accumulate the total initial sum of lengths of all teeth
    initial_total_sum = 0
    # Variable to store the minimum sum U_i + D_i over all pairs. This serves as an upper bound for H.
    # Initialize with infinity to ensure any sum will be smaller.
    min_S = float('inf') 
    
    # Read N pairs of tooth lengths
    for _ in range(N):
        # Read U_i and D_i for the current pair
        ui, di = map(int, sys.stdin.readline().split())
        # Store the pair information
        pairs.append({'U': ui, 'D': di})
        # Add the sum of lengths of this pair to the total initial sum
        initial_total_sum += ui + di
        # Update the minimum sum found so far
        min_S = min(min_S, ui + di)

    # `check(H)` function determines if a given target sum H is feasible.
    # A value H is feasible if there exists a sequence of final upper tooth lengths U'_1, ..., U'_N such that:
    # 1. For each i, U'_i + D'_i = H, where D'_i is the final length of the i-th lower tooth.
    # 2. Constraints from grinding: 0 <= U'_i <= U_i and 0 <= D'_i <= D_i for all i.
    #    These translate to bounds on U'_i: max(0, H - D_i) <= U'_i <= min(U_i, H).
    # 3. Adjacency constraint: |U'_i - U'_{i+1}| <= X for all 1 <= i < N.
    def check(H):
        # H must be non-negative. Although binary search starts from 0, this check is included for clarity.
        if H < 0: 
             return False

        # Calculate initial bounds [L_i, R_i] for each potential final upper tooth length U'_i based on H.
        Ls = [] # List to store lower bounds for U'_i
        Rs = [] # List to store upper bounds for U'_i
        for i in range(N):
            ui = pairs[i]['U'] # Initial length of i-th upper tooth
            di = pairs[i]['D'] # Initial length of i-th lower tooth
            
            # Calculate the lower bound for U'_i.
            # From D'_i <= D_i and D'_i = H - U'_i, we get H - U'_i <= D_i => U'_i >= H - D_i.
            # Also, final lengths must be non-negative, so U'_i >= 0.
            # Combining these, li = max(0, H - D_i).
            li = max(0, H - di)
            
            # Calculate the upper bound for U'_i.
            # From U'_i <= U_i.
            # Also, from D'_i >= 0 and D'_i = H - U'_i, we get H - U'_i >= 0 => U'_i <= H.
            # Combining these, ri = min(U_i, H).
            ri = min(ui, H)
            
            # If for any tooth pair the calculated lower bound exceeds the upper bound,
            # it's impossible to satisfy the constraints for this H, so H is not feasible.
            if li > ri:
                return False
            Ls.append(li)
            Rs.append(ri)

        # Propagate the adjacency constraint |U'_i - U'_{i+1}| <= X through the sequence of teeth.
        # We use two passes: one forward (left-to-right) and one backward (right-to-left)
        # to determine the tightest possible range for each U'_i satisfying all constraints.

        # Forward pass: Calculate range [a_i, b_i] for U'_i considering constraints from U'_1 to U'_i.
        a = [0] * N  # a_i stores the minimum possible value for U'_i based on left constraints
        b = [0] * N  # b_i stores the maximum possible value for U'_i based on left constraints
        a[0] = Ls[0] # Base case for U'_1: range is initially [L_1, R_1]
        b[0] = Rs[0]
        
        # Check if the base case range is valid
        if a[0] > b[0]:
             return False 

        # Iterate from i=0 to N-2 to compute ranges for U'_{i+1} based on U'_i
        for i in range(N - 1):
            # Lower bound for U'_{i+1}: Must satisfy U'_{i+1} >= Ls[i+1] AND U'_{i+1} >= U'_i - X.
            # Since U'_i >= a[i], we must have U'_{i+1} >= a[i] - X. Combine these: max(Ls[i+1], a[i] - X).
            a[i+1] = max(Ls[i+1], a[i] - X)
            # Upper bound for U'_{i+1}: Must satisfy U'_{i+1} <= Rs[i+1] AND U'_{i+1} <= U'_i + X.
            # Since U'_i <= b[i], we must have U'_{i+1} <= b[i] + X. Combine these: min(Rs[i+1], b[i] + X).
            b[i+1] = min(Rs[i+1], b[i] + X)
            # If at any step the calculated range [a_{i+1}, b_{i+1}] is empty (lower > upper), H is not feasible.
            if a[i+1] > b[i+1]:
                return False

        # Backward pass: Calculate range [c_i, d_i] for U'_i considering constraints from U'_N down to U'_i.
        c = [0] * N # c_i stores minimum possible value for U'_i based on right constraints
        d = [0] * N # d_i stores maximum possible value for U'_i based on right constraints
        c[N-1] = Ls[N-1] # Base case for U'_N: range is initially [L_N, R_N]
        d[N-1] = Rs[N-1]
        
        # Check if the base case range is valid
        if c[N-1] > d[N-1]:
             return False

        # Iterate from i=N-1 down to 1 to compute ranges for U'_{i-1} based on U'_i
        for i in range(N - 1, 0, -1):
            # Lower bound for U'_{i-1}: Must satisfy U'_{i-1} >= Ls[i-1] AND U'_{i-1} >= U'_i - X.
            # Since U'_i >= c[i], we must have U'_{i-1} >= c[i] - X. Combine these: max(Ls[i-1], c[i] - X).
            c[i-1] = max(Ls[i-1], c[i] - X)
            # Upper bound for U'_{i-1}: Must satisfy U'_{i-1} <= Rs[i-1] AND U'_{i-1} <= U'_i + X.
            # Since U'_i <= d[i], we must have U'_{i-1} <= d[i] + X. Combine these: min(Rs[i-1], d[i] + X).
            d[i-1] = min(Rs[i-1], d[i] + X)
            # If derived range is empty, H is not feasible.
            if c[i-1] > d[i-1]:
                return False

        # Final consistency check: For a feasible H, there must exist a valid value for each U'_i.
        # The true feasible range for U'_i is the intersection of the range from the forward pass [a_i, b_i]
        # and the range from the backward pass [c_i, d_i]. This intersection must be non-empty for all i.
        for i in range(N):
            # Calculate the intersection interval [final_L, final_R]
            # The final lower bound is the maximum of the lower bounds from both passes.
            final_L = max(a[i], c[i])
            # The final upper bound is the minimum of the upper bounds from both passes.
            final_R = min(b[i], d[i])
            # If the intersection is empty (lower bound > upper bound), then no value for U'_i exists, so H is not feasible.
            if final_L > final_R:
                return False
        
        # If all checks passed successfully, it means H is feasible.
        return True

    # Binary search for the maximum feasible value of H.
    # The possible range for H is [0, min_S]. H=0 is always feasible.
    # We perform binary search on this range. `high` is used as an exclusive upper bound.
    low = 0          # Minimum possible value for H
    high = min_S + 1 # Exclusive upper bound for H
    
    # Store the best feasible H found so far. Initialized to 0.
    optimal_H = 0 

    # Standard binary search loop
    while low < high:
        # Calculate midpoint. Using integer division //
        mid = (low + high) // 2
        # Check if the midpoint H value is feasible
        if check(mid):
            # If `mid` is feasible, it could be the maximum feasible H, or a larger H might also be feasible.
            # Store `mid` as the current best candidate and try searching in the upper half: [mid+1, high).
            optimal_H = mid
            low = mid + 1
        else:
            # If `mid` is not feasible, the maximum feasible H must be smaller than `mid`.
            # Search in the lower half: [low, mid). Update `high` to `mid`.
            high = mid

    # After the binary search terminates (low == high), `optimal_H` holds the maximum feasible value for H.
    # The minimum cost required is the difference between the total initial length of all teeth
    # and the total final length of all teeth.
    # The total final length sum is N * optimal_H because U'_i + D'_i = H for all i.
    min_cost = initial_total_sum - N * optimal_H
    # Print the calculated minimum cost.
    print(min_cost)

# Execute the solve function to run the program
solve()