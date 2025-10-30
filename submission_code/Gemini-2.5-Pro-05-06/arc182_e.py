import math
import bisect

def gcd(a, b):
    return math.gcd(a, b)

def solve():
    N, M, C, K = map(int, input().split())
    A_orig = list(map(int, input().split()))

    if C == 0:
        min_A = A_orig[0]
        for i in range(1, N):
            if A_orig[i] < min_A:
                min_A = A_orig[i]
        print(K * min_A)
        return

    g = gcd(C, M)
    C0 = C // g
    M0 = M // g

    r_min = -1
    if N > 0: # N >= 1 constraint
        r_min = A_orig[0] % g
        for i in range(1, N):
            val_mod_g = A_orig[i] % g
            if val_mod_g < r_min:
                r_min = val_mod_g
    
    A0_list = [] # Changed name from A0 to avoid confusion with A_uni[0]
    for x_val in A_orig:
        if x_val % g == r_min:
            A0_list.append(x_val // g)
    
    # If A0_list is empty, it means something went wrong or N=0.
    # Given N>=1, A0_list cannot be empty. r_min is selected from A_orig[i]%g.
    # So at least one element A_orig[i] will satisfy A_orig[i]%g == r_min.
    
    A_uni = sorted(list(set(A0_list)))
    D = len(A_uni)

    S_prime_period = 0
    # D can be 0 if A0_list was empty. If A0_list is non-empty, D >= 1.
    # The problem implies N >= 1. So A_orig has at least one element.
    # r_min will be found. A0_list will have at least one element. So D >= 1.

    # Part 1: y in [0, A_uni[0]]
    # h(y) = A_uni[0] - y. Sum is sum_{t=0}^{A_uni[0]} t
    val = A_uni[0]
    S_prime_period += val * (val + 1) // 2
    
    # Part 2: y in (A_uni[j], A_uni[j+1]]
    # h(y) = A_uni[j+1] - y. Sum is sum_{t=0}^{L-1} t where L = A_uni[j+1]-A_uni[j]
    for j in range(D - 1):
        len_segment = A_uni[j+1] - A_uni[j]
        if len_segment > 0:
            S_prime_period += (len_segment - 1) * len_segment // 2
    
    # Part 3: y in (A_uni[D-1], M0-1]
    # h(y) = (A_uni[0] + M0) - y.
    # y ranges from A_uni[D-1]+1 to M0-1
    # The values h(y) take are (A_uni[0]+M0)-(A_uni[D-1]+1), ..., (A_uni[0]+M0)-(M0-1)
    # which is an arithmetic progression: last_sum_term, ..., first_sum_term
    
    # Number of terms for y in this range
    num_y_terms_in_interval = (M0 - 1) - (A_uni[D-1] + 1) + 1 
    if num_y_terms_in_interval > 0:
        # Smallest value h(y) takes in this interval (when y is M0-1)
        first_h_val = (A_uni[0] + M0) - (M0 - 1) 
        # Largest value h(y) takes in this interval (when y is A_uni[D-1]+1)
        last_h_val = (A_uni[0] + M0) - (A_uni[D-1] + 1)
        S_prime_period += num_y_terms_in_interval * (first_h_val + last_h_val) // 2
    
    S_period_one_cycle = g * S_prime_period + M0 * r_min
    
    num_full_periods = K // M0
    total_sum = num_full_periods * S_period_one_cycle
    
    K_rem = K % M0
    
    if K_rem > 0:
        S_prime_rem = 0
        
        y_step = (-C0 % M0 + M0) % M0 
        current_y_for_h = 0 # This is y_k in my thoughts

        for _ in range(K_rem):
            # current_y_for_h is argument to h()
            # Find which segment current_y_for_h falls into to calculate h(current_y_for_h)
            
            # idx is such that A_uni[idx-1] < current_y_for_h <= A_uni[idx]
            # Or current_y_for_h <= A_uni[0]
            # Or current_y_for_h > A_uni[D-1]
            idx = bisect.bisect_left(A_uni, current_y_for_h)

            val_h_y = 0
            if idx == 0: # current_y_for_h <= A_uni[0]
                val_h_y = A_uni[0] - current_y_for_h
            elif idx == D: # current_y_for_h > A_uni[D-1] (and current_y_for_h < M0)
                val_h_y = (A_uni[0] + M0) - current_y_for_h
            else: # A_uni[idx-1] < current_y_for_h <= A_uni[idx]
                val_h_y = A_uni[idx] - current_y_for_h
            
            S_prime_rem += val_h_y
            current_y_for_h = (current_y_for_h + y_step) % M0 # Next y_k

        S_rem_contrib = g * S_prime_rem + K_rem * r_min
        total_sum += S_rem_contrib
        
    print(total_sum)

solve()