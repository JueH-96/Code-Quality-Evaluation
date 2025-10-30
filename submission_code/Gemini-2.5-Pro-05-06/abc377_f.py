import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    # No pieces given in input, only their count M.
    # Need to read M lines for piece coordinates.
    # pieces_coords = [] # Not strictly needed to store all coords if processed one by one for sets
    
    R_threat = set()
    C_threat = set()
    D1_threat = set() # r - c
    D2_threat = set() # r + c

    for _ in range(M):
        r, c = map(int, sys.stdin.readline().split())
        # pieces_coords.append((r,c))
        R_threat.add(r)
        C_threat.add(c)
        D1_threat.add(r - c)
        D2_threat.add(r + c)

    num_safe_rows = N - len(R_threat)
    num_safe_cols = N - len(C_threat)

    # Initial count: squares (r,c) where r is a safe row and c is a safe column
    ans = num_safe_rows * num_safe_cols

    # Term 1: Subtract |{(r,c) in S_R x S_C | r-c in D1_threat}|
    # This is sum over d1 in D1_threat of count of r s.t.
    # r in S_R (not in R_threat), (r-d1) in S_C (not in C_threat)
    # and r, (r-d1) are valid coordinates.
    
    sum_term1 = 0
    for d1 in D1_threat:
        # For a fixed d1, c = r-d1.
        # Conditions for r:
        # 1 <= r <= N
        # 1 <= r-d1 <= N  =>  d1+1 <= r <= N+d1
        # So, r_min = max(1, 1 + d1) and r_max = min(N, N + d1)
        r_min = max(1, 1 + d1)
        r_max = min(N, N + d1)

        if r_min > r_max: # No valid r in this range
            continue

        # Number of integers r in [r_min, r_max]
        count_r_total_in_range = r_max - r_min + 1
        
        # We need r NOT IN R_threat AND (r-d1) NOT IN C_threat.
        # (r-d1) IN C_threat is equivalent to r IN { ct + d1 | ct in C_threat }.
        # So, we count r such that r IS IN R_threat OR r IS IN { ct + d1 | ct in C_threat }.
        # These are "bad" r values.
        
        bad_r_values_in_range = set()
        # Check r values that are bad because r is in R_threat
        for rt_val in R_threat:
            if r_min <= rt_val <= r_max:
                bad_r_values_in_range.add(rt_val)
        
        # Check r values that are bad because (r-d1) is in C_threat
        for ct_val in C_threat:
            r_candidate_from_c = ct_val + d1
            if r_min <= r_candidate_from_c <= r_max:
                bad_r_values_in_range.add(r_candidate_from_c)
        
        # Number of valid r for this d1
        count_r_for_d1_valid = count_r_total_in_range - len(bad_r_values_in_range)
        sum_term1 += count_r_for_d1_valid
    
    ans -= sum_term1

    # Term 2: Subtract |{(r,c) in S_R x S_C | r+c in D2_threat}|
    # Similar logic, but iterate over c for fixed d2.
    
    sum_term2 = 0
    for d2 in D2_threat:
        # For a fixed d2, r = d2-c.
        # Conditions for c:
        # 1 <= c <= N
        # 1 <= d2-c <= N  =>  d2-N <= c <= d2-1
        # So, c_min = max(1, d2-N) and c_max = min(N, d2-1)
        c_min = max(1, d2 - N)
        c_max = min(N, d2 - 1)

        if c_min > c_max: # No valid c in this range
            continue
            
        count_c_total_in_range = c_max - c_min + 1

        # We need c NOT IN C_threat AND (d2-c) NOT IN R_threat.
        # (d2-c) IN R_threat is equivalent to c IN { d2 - rt | rt in R_threat }.
        # So, count c such that c IS IN C_threat OR c IS IN { d2 - rt | rt in R_threat }.
        
        bad_c_values_in_range = set()
        # Check c values that are bad because c is in C_threat
        for ct_val in C_threat:
            if c_min <= ct_val <= c_max:
                bad_c_values_in_range.add(ct_val)
        
        # Check c values that are bad because (d2-c) is in R_threat
        for rt_val in R_threat:
            c_candidate_from_r = d2 - rt_val
            if c_min <= c_candidate_from_r <= c_max:
                bad_c_values_in_range.add(c_candidate_from_r)
        
        count_c_for_d2_valid = count_c_total_in_range - len(bad_c_values_in_range)
        sum_term2 += count_c_for_d2_valid
        
    ans -= sum_term2

    # Term 3: Add |{(r,c) in S_R x S_C | r-c in D1_threat AND r+c in D2_threat}|
    # Iterate over d1 in D1_threat and d2 in D2_threat.
    # For each pair (d1, d2), find unique (r,c):
    # r = (d1+d2)/2, c = (d2-d1)/2.
    # Check if this (r,c) is valid and r not in R_threat, c not in C_threat.
    
    sum_term3 = 0
    for d1 in D1_threat:
        for d2 in D2_threat:
            if (d1 + d2) % 2 != 0: # r must be an integer
                continue
            # If (d1+d2) is even, d1 and d2 have same parity, so (d2-d1) is also even.
            # Thus c will also be an integer.
            
            r_cand = (d1 + d2) // 2
            c_cand = (d2 - d1) // 2

            # Check if (r_cand, c_cand) is within grid boundaries
            if not (1 <= r_cand <= N and 1 <= c_cand <= N):
                continue
            
            # Check if r_cand is a safe row and c_cand is a safe column
            if r_cand not in R_threat and c_cand not in C_threat:
                sum_term3 += 1
                
    ans += sum_term3
    
    sys.stdout.write(str(ans) + "
")

solve()