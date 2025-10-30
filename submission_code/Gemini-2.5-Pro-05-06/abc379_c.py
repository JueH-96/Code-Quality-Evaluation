import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    if M == 0:
        # N >= 2 is a problem constraint.
        # If M=0, there are no stones. If N > 0, impossible to fill N cells.
        print("-1")
        return

    X_coords_input = list(map(int, sys.stdin.readline().split()))
    A_counts_input = list(map(int, sys.stdin.readline().split()))
    
    stones_data = []
    for i in range(M):
        stones_data.append((X_coords_input[i], A_counts_input[i]))
    
    # Sort by X_i. Problem states X_i are distinct.
    stones_data.sort() 
    
    sum_AX = 0
    # Calculate sum_AX using the (now sorted) stones_data.
    # Summation is commutative, so sum_AX is the same regardless of order.
    for x_coord, a_count in stones_data:
        sum_AX += x_coord * a_count
        
    # P_cur stores the current P_k value, where P_k = sum_{j=1 to k} (A'_j - 1)
    # A'_j is the number of stones initially at cell j.
    # x_prev stores the X-coordinate of the previously processed pile of stones,
    # or 0 if no piles processed yet (representing conceptual cell 0 where P_0 = 0).
    
    P_cur = 0
    x_prev = 0
    
    for i in range(M):
        x, a = stones_data[i] # Current pile at x with a stones
        
        # Calculate length of segment of cells without initial stones: (x_prev, x).
        # These are cells x_prev+1, ..., x-1.
        # Number of such cells = (x-1) - (x_prev+1) + 1 = x - x_prev - 1.
        gap_len = x - x_prev - 1
        
        # P_cur currently holds P_{x_prev}.
        # For cells k in (x_prev, x), P_k decreases by 1 for each step.
        # The minimum P_k in this segment (before cell x) is P_{x-1}.
        # P_{x-1} = P_{x_prev} - ((x-1) - x_prev) = P_{x_prev} - gap_len.
        # For P_{x-1} to be non-negative, P_{x_prev} must be >= gap_len.
        if P_cur < gap_len:
            print("-1")
            return
        
        P_cur -= gap_len # P_cur is now P_{x-1}
        
        # At cell x, A'_x = a. So P_x = P_{x-1} + (A'_x - 1) = P_{x-1} + (a - 1).
        P_cur += (a - 1) # P_cur is now P_x
        
        # Since A_i >= 1 (problem constraint), a-1 >= 0.
        # If P_{x-1} >= 0 (which was ensured by the P_cur < gap_len check,
        # as P_cur after subtraction becomes P_{x-1} and must be non-negative),
        # then P_x will also be >= 0.
        # So, an explicit check `if P_cur < 0:` here is redundant.
            
        x_prev = x # Update x_prev for the next iteration
        
    # After processing all M piles, x_prev is the location of the M-th pile (last pile X_M).
    # P_cur holds P_{x_prev} (i.e., P_{X_M}).
    # Consider cells from x_prev + 1 to N. These are initially empty.
    # Number of such cells = N - (x_prev+1) + 1 = N - x_prev.
    gap_len_tail = N - x_prev
    
    # Similar to the loop, P_{x_prev} must be >= gap_len_tail for P_N to be non-negative.
    # (Specifically, P_N must be exactly 0 for possibility).
    # P_N = P_{x_prev} - gap_len_tail.
    if P_cur < gap_len_tail: # This means P_N would be < 0.
        print("-1")
        return
        
    P_cur -= gap_len_tail # P_cur is now P_N
    
    # Final check: P_N must be exactly 0.
    # P_N = (sum of all A_i_initial) - N. So P_N = 0 iff sum A_i_initial = N.
    if P_cur != 0:
        print("-1")
        return
        
    # If all checks pass, it's possible.
    # Minimum operations = (sum of final positions) - (sum of initial positions)
    # Sum of final positions = 1 + 2 + ... + N = N * (N + 1) // 2
    # Sum of initial positions = sum_AX (calculated earlier)
    
    total_ops = N * (N + 1) // 2 - sum_AX
    print(total_ops)

solve()