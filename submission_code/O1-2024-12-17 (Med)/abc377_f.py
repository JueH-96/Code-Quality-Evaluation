def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parse
    N = int(input_data[0])
    M = int(input_data[1])
    pieces = [(int(input_data[2+2*k]), int(input_data[3+2*k])) for k in range(M)]
    
    # Collect dangerous rows, cols, main-diagonals (i+j), anti-diagonals (i-j)
    R_set = set()
    C_set = set()
    Dp_set = set()  # i+j
    Dm_set = set()  # i-j
    
    for (r, c) in pieces:
        R_set.add(r)
        C_set.add(c)
        Dp_set.add(r + c)
        Dm_set.add(r - c)
    
    R_list = sorted(R_set)
    C_list = sorted(C_set)
    Dp_list = sorted(Dp_set)
    Dm_list = sorted(Dm_set)

    # Size of safe rows/cols if we only consider row/column danger
    Sr_size = N - len(R_set)  # number of rows not in R_set
    Sc_size = N - len(C_set)  # number of columns not in C_set
    
    # If there are no safe rows or no safe columns, answer is trivially 0
    if Sr_size <= 0 or Sc_size <= 0:
        # But we still must subtract squares that are already occupied (though result is obviously 0).
        print(0)
        return

    # Utility: count how many elements of sorted_arr lie in [low, high] using bisect.
    import bisect
    def count_in_segment(sorted_arr, low, high):
        # number of elements x in sorted_arr with low <= x <= high
        left_index = bisect.bisect_left(sorted_arr, low)
        right_index = bisect.bisect_right(sorted_arr, high)
        return right_index - left_index

    # Count how many (i, j) in the "row/column safe region" (i not in R, j not in C)
    # fall on a main-diagonal in Dp. Denote that set as S1.
    # For each d in Dp, i+j=d => i in [max(1, d-N) .. min(N, d-1)].
    # We exclude i in R and j=d-i in C via a small inclusion-exclusion.
    def count_S1():
        ans = 0
        for d in Dp_list:
            L = max(1, d - N)
            U = min(N, d - 1)
            if L > U:
                continue
            length = U - L + 1
            # number of i in R within [L,U]
            bad_r = count_in_segment(R_list, L, U)
            # build set of i_c = d - c for c in C, within [L,U]
            # that correspond to j = c => j in C => exclude
            i_c_vals = []
            for c in C_list:
                i_c = d - c
                if L <= i_c <= U:
                    i_c_vals.append(i_c)
            size_ic = len(i_c_vals)
            if size_ic == 0 and bad_r == 0:
                # Quick path: no overlaps, so everything in [L..U] is valid
                ans += length
                continue
            
            i_c_set = set(i_c_vals)
            # overlap = how many i in R intersect i_c_set
            overlap = 0
            # Count how many r in R_list that lie in [L,U] are also in i_c_set
            left_idx = bisect.bisect_left(R_list, L)
            right_idx = bisect.bisect_right(R_list, U)
            for idx in range(left_idx, right_idx):
                r = R_list[idx]
                if r in i_c_set:
                    overlap += 1
            
            ans += (length - bad_r - size_ic + overlap)
        return ans

    # Similarly, count how many (i, j) in the "row/column safe region" fall on an anti-diagonal in Dm.
    # i-j=d => j=i-d => i in [max(1, d+1).. min(N, N+d]] if that is valid.
    def count_S2():
        ans = 0
        for d in Dm_list:
            L = max(1, d + 1)
            U = min(N, N + d)
            if L > U:
                continue
            length = U - L + 1
            # exclude i in R
            bad_r = count_in_segment(R_list, L, U)
            # exclude j in C => j = i-d => i = j + d => so i != x for x in {d + c | c in C}
            i_c_vals = []
            for c in C_list:
                i_c = d + c
                if L <= i_c <= U:
                    i_c_vals.append(i_c)
            size_ic = len(i_c_vals)
            if size_ic == 0 and bad_r == 0:
                ans += length
                continue
            i_c_set = set(i_c_vals)
            overlap = 0
            left_idx = bisect.bisect_left(R_list, L)
            right_idx = bisect.bisect_right(R_list, U)
            for idx in range(left_idx, right_idx):
                r = R_list[idx]
                if r in i_c_set:
                    overlap += 1
            ans += (length - bad_r - size_ic + overlap)
        return ans

    S1 = count_S1()
    S2 = count_S2()

    # Now count S1 ∩ S2.  For (i, j) in that intersection, we must have
    # i+j in Dp and i-j in Dm, plus i not in R, j not in C, and within bounds.
    # Solve i = ( (i+j) + (i-j) )/2 = (d1 + d2)/2, j = (d1 - d2)/2 for d1 in Dp, d2 in Dm.
    # We do up to len(Dp)*len(Dm) <= 1000*1000 = 1e6.
    R_set_local = R_set  # for fast lookup
    C_set_local = C_set
    count_inter = 0
    for d1 in Dp_list:
        for d2 in Dm_list:
            # They must have same parity for i, j to be integers
            # (d1 + d2) must be even => d1 %2 == d2 %2
            # We can just check if (d1 + d2) %2 == 0
            if (d1 + d2) & 1:  # odd
                continue
            i = (d1 + d2)//2
            j = (d1 - d2)//2
            if i < 1 or i > N or j < 1 or j > N:
                continue
            if i not in R_set_local and j not in C_set_local:
                count_inter += 1

    # Number of squares "blocked" by diagonals among row/column-safe squares
    blocked = S1 + S2 - count_inter
    # So far, the number of squares that are safe from row/col/diagonal captures:
    safe_squares = (Sr_size * Sc_size) - blocked

    # Finally, we must exclude squares that are already occupied by existing pieces,
    # but only those that would have been counted as safe. Check each existing piece:
    occupied_safe = 0
    R_set_lookup = R_set
    C_set_lookup = C_set
    Dp_set_lookup = Dp_set
    Dm_set_lookup = Dm_set
    for (r, c) in pieces:
        if r not in R_set_lookup and c not in C_set_lookup:
            if (r + c) not in Dp_set_lookup and (r - c) not in Dm_set_lookup:
                occupied_safe += 1

    answer = safe_squares - occupied_safe
    # In principle, this should not go negative, but just in case:
    if answer < 0:
        answer = 0
    
    print(answer)