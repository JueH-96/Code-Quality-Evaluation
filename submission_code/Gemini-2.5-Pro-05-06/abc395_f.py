import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())
    
    U_orig = []
    D_orig = []
    
    sum_S = 0
    # Calculate min_S_val = min(U_i + D_i) over all i.
    # Max possible H is min_S_val. Smallest possible U_i+D_i is 1+1=2.
    # float('inf') is a safe initial value.
    min_S_val = float('inf') 
    
    for _ in range(N):
        u, d = map(int, sys.stdin.readline().split())
        U_orig.append(u)
        D_orig.append(d)
        s_i = u + d
        sum_S += s_i
        min_S_val = min(min_S_val, s_i)

    # check(target_H) returns True if it's possible to satisfy conditions with this H
    def check(target_H):
        # current_dp_l, current_dp_r will store the valid range for U_i'
        # (using 0-indexed arrays for U_orig, D_orig)
        
        # For U_0'
        # Its intrinsic range is [L_0, R_0]
        # L_0 = max(0, target_H - D_orig[0])
        # R_0 = U_orig[0]
        current_dp_l = max(0, target_H - D_orig[0])
        current_dp_r = U_orig[0]
        
        # If L_0 > R_0, target_H is impossible.
        # This happens if target_H > U_orig[0] + D_orig[0].
        # This case is technically ruled out if binary search upper bound is min_S_val.
        # However, an explicit check is safer and makes check() more general.
        if current_dp_l > current_dp_r: 
            return False

        # For U_1' to U_{N-1}'
        for i in range(1, N):
            # Intrinsic range for U_i' is [Li, Ri]
            Li = max(0, target_H - D_orig[i])
            Ri = U_orig[i]
            
            if Li > Ri: # target_H > U_orig[i] + D_orig[i]
                return False

            # Range for U_i' based on U_{i-1}' (which was in [current_dp_l, current_dp_r])
            # and constraint X is [current_dp_l - X, current_dp_r + X].
            prev_constraint_L = current_dp_l - X
            prev_constraint_R = current_dp_r + X
            
            # Actual interval for U_i' is the intersection of [Li, Ri] and [prev_constraint_L, prev_constraint_R]
            current_dp_l = max(Li, prev_constraint_L)
            current_dp_r = min(Ri, prev_constraint_R)
            
            if current_dp_l > current_dp_r: # Intersection is empty
                return False
        
        return True # Possible to find such U_i' sequence

    # Binary search for the maximum H
    # Search space for H is [0, min_S_val].
    # best_H = 0 initially: corresponds to choosing all U_i'=0, D_i'=0.
    # This is always possible if X >= 0 (given X >= 1). Cost would be sum_S.
    low = 0
    high = min_S_val 
    best_H = 0 
    
    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            best_H = mid
            low = mid + 1 # Try for a larger H
        else:
            high = mid - 1 # mid is too high or conditions for Li>Ri met
            
    min_cost = sum_S - N * best_H
    print(min_cost)

solve()