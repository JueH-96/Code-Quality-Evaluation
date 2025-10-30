import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())
    
    U_orig = [0] * N
    D_orig = [0] * N
    sum_S_orig = 0 # sum of U_i_orig + D_i_orig for cost calculation
    
    for i in range(N):
        u, d = map(int, sys.stdin.readline().split())
        U_orig[i] = u
        D_orig[i] = d
        sum_S_orig += (u + d)

    # Function to check if a given H (target sum U_i_final + D_i_final) is feasible.
    # It checks if there exists a sequence U_1_final, ..., U_N_final such that:
    # 1. max(0, H - D_orig[i]) <= U_i_final <= min(U_orig[i], H) for all i
    # 2. |U_i_final - U_{i+1}_final| <= X for all i
    def check(H):
        # current_min_u and current_max_u track the feasible range for U_i_final
        # as we propagate constraints from left to right.
        
        # Calculate the initial L_0 and R_0 for the first tooth (index 0)
        # L_0: U_0_final must be at least 0 AND at least H - D_0_orig (to ensure D_0_final <= D_0_orig)
        current_min_u = max(0, H - D_orig[0])
        # R_0: U_0_final must be at most U_0_orig AND at most H (to ensure D_0_final >= 0)
        current_max_u = min(U_orig[0], H)

        # If the initial range for U_0_final is invalid, H is not feasible
        if current_min_u > current_max_u:
            return False

        # Propagate constraints for subsequent teeth
        for i in range(1, N):
            # Calculate L_i and R_i for the current tooth (index i)
            li = max(0, H - D_orig[i])
            ri = min(U_orig[i], H)

            # If the individual range [li, ri] for current tooth is invalid, H is not feasible
            if li > ri:
                return False

            # Calculate the allowed range for U_i_final based on U_{i-1}_final's range
            # and the |U_i - U_{i+1}| <= X constraint.
            # U_i_final must be in [current_min_u_prev - X, current_max_u_prev + X]
            next_min_u_from_prev = current_min_u - X
            next_max_u_from_prev = current_max_u + X

            # Intersect this range with the current tooth's intrinsic range [li, ri]
            new_min_u = max(li, next_min_u_from_prev)
            new_max_u = min(ri, next_max_u_from_prev)

            # If the intersected range is empty, H is not feasible
            if new_min_u > new_max_u:
                return False

            # Update the feasible range for the current tooth
            current_min_u = new_min_u
            current_max_u = new_max_u
        
        # If we successfully propagated feasible ranges through all teeth, H is feasible
        return True

    # Binary search for the maximum feasible H
    low = 0
    # The maximum possible sum for a pair (U_i_orig + D_i_orig) is 10^9 + 10^9 = 2 * 10^9.
    # Thus, H cannot exceed 2 * 10^9. A slightly larger upper bound for 'high' is safe.
    high = 2 * (10**9) + 7 
    
    ans_H = 0 # Stores the maximum H for which check(H) returns True

    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            ans_H = mid         # mid_H is feasible, store it and try for a larger H
            low = mid + 1
        else:
            high = mid - 1      # mid_H is not feasible, need a smaller H
    
    # The minimum cost is the total initial sum minus N times the maximized feasible H.
    # Python handles large integers automatically, so no overflow issues for sum_S_orig or min_cost.
    min_cost = sum_S_orig - N * ans_H
    sys.stdout.write(str(min_cost) + '
')

# Call the solve function to run the program
solve()