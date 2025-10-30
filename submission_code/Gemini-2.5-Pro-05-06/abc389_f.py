import sys

# Adjust recursion limit for segment tree construction if N is large
sys.setrecursionlimit(4 * 10**5) 

def solve():
    N = int(sys.stdin.readline())
    contests = []
    for _ in range(N):
        contests.append(list(map(int, sys.stdin.readline().split())))

    Q_count = int(sys.stdin.readline())
    queries_input = []
    max_initial_X = 0
    for i in range(Q_count):
        x = int(sys.stdin.readline())
        queries_input.append((x, i))
        if x > max_initial_X:
            max_initial_X = x
    
    if N == 0: # No contests, rating doesn't change
        results = [0] * Q_count
        for x_val, original_idx in queries_input:
            results[original_idx] = x_val
        for res in results:
            sys.stdout.write(str(res) + "
")
        return

    MIN_COORD = 1
    # Max rating can be max_initial_X + N or max_R_val + N.
    # max_R_val from contests can be up to 5e5.
    # Overall max coordinate needed for function evaluation.
    # Max initial X query could be 5e5. Max N is 2e5. Sum = 7e5.
    # Max R_i is 5e5. Max N is 2e5. Sum = 7e5.
    MAX_COORD = max(max_initial_X, 500000) + N + 5 # Add a small buffer

    # --- Function representation and operations ---

    memo_eval_func = {} # Cache for eval_func results
    def eval_func(func_pts_tuple, x_query):
        # func_pts_tuple is a tuple of (x, y) tuples, sorted by x
        # Using tuple makes it hashable for memoization
        
        # Memoization check - func_pts_tuple is hashable
        state = (func_pts_tuple, x_query)
        if state in memo_eval_func:
            return memo_eval_func[state]

        func_pts = list(func_pts_tuple) # Work with list version

        low = 0
        high = len(func_pts) - 1
        seg_idx = -1

        while low <= high:
            mid = (low + high) // 2
            if func_pts[mid][0] <= x_query:
                seg_idx = mid
                low = mid + 1
            else:
                high = mid - 1
        
        res = 0
        if seg_idx == -1: # x_query < func_pts[0][0]
            # Extrapolate left with slope 1 (implicit from problem structure)
            res = func_pts[0][1] + (x_query - func_pts[0][0])
        elif seg_idx == len(func_pts) - 1: # x_query >= func_pts[last_idx][0]
            # Extrapolate right with slope 1
            res = func_pts[seg_idx][1] + (x_query - func_pts[seg_idx][0])
        else:
            x0, y0 = func_pts[seg_idx]
            x1, y1 = func_pts[seg_idx+1]

            if x0 == x1: # Should not happen with proper cleaning (vertical segment)
                res = y0 
            else: # Linear interpolation
                # Slope is guaranteed to be integer for this problem's functions.
                # Or rather, (y1-y0) must be divisible by (x1-x0) if slope is not int,
                # such that (y1-y0)//(x1-x0) * (x_query-x0) gives correct integer offset.
                # All points are integers, all operations X+1. So F(X) is integer.
                # Slope calculation: (y1-y0) / (x1-x0)
                # val = y0 + slope * (x_query - x0)
                res = y0 + (y1 - y0) * (x_query - x0) // (x1 - x0)
        
        memo_eval_func[state] = res
        return res

    def eval_inverse_func(pts_list_for_inverse, segment_start_idx, target_y):
        # pts_list_for_inverse is a list of (x,y) points
        idx = segment_start_idx
        x0, y0 = pts_list_for_inverse[idx]
        x1, y1 = pts_list_for_inverse[idx+1]

        if y0 == y1: # Horizontal segment
            return x0 if target_y == y0 else None
        
        if not ((min(y0, y1) <= target_y <= max(y0, y1))): # Target_y outside segment's y-range
             return None

        # Inverse linear interpolation: x = x0 + (x1-x0)/(y1-y0) * (target_y - y0)
        # All points are integers, resulting x must be integer.
        # This implies (x1-x0) * (target_y-y0) must be divisible by (y1-y0).
        if ( (x1 - x0) * (target_y - y0) % (y1 - y0) ) != 0:
            return None # Resulting x is not integer, so not a breakpoint from this path

        val = x0 + (x1 - x0) * (target_y - y0) // (y1 - y0)
        
        # Check if val is within the original x-segment [x0, x1]
        if not (min(x0,x1) <= val <= max(x0,x1)): # Should be covered by y-range check if monotonic
            return None
        return val

    def clean_func_pts(pts_list):
        if not pts_list: return tuple()
        
        # Ensure sorted by x, and unique x (prefer later y for same x, though not expected)
        # The generation logic should give sorted unique x.
        # For now, assume pts_list is already sorted by x and x are unique.
        
        cleaned_pts = [pts_list[0]]
        for i in range(1, len(pts_list) - 1):
            x_prev, y_prev = cleaned_pts[-1]
            x_curr, y_curr = pts_list[i]
            x_next, y_next = pts_list[i+1]
            
            if x_curr == x_prev : # Should not happen if x are unique
                cleaned_pts[-1] = (x_curr, y_curr) # Update y; take the latter point's y
                continue

            # Check for collinearity (integer cross product)
            val1 = (y_curr - y_prev) * (x_next - x_curr)
            val2 = (y_next - y_curr) * (x_curr - x_prev)
            if val1 != val2: 
                cleaned_pts.append((x_curr,y_curr))
        
        if len(pts_list) > 1:
            # Ensure last point is added and handles if it's same as current last cleaned one
            if cleaned_pts[-1][0] == pts_list[-1][0]: # Same X as last cleaned point
                 if cleaned_pts[-1][1] != pts_list[-1][1]: # If Y different, update (should mean vertical line end)
                     cleaned_pts[-1] = pts_list[-1] # Not expected for these functions
            else: # Different X
                cleaned_pts.append(pts_list[-1])
        return tuple(cleaned_pts)


    def compose_funcs(func_B_pts_tuple, func_A_pts_tuple):
        func_A_pts = list(func_A_pts_tuple) # Convert to list for indexing
        func_B_pts = list(func_B_pts_tuple)

        composed_raw_pts = []
        
        ptr_A = 0
        ptr_B = 0
        
        current_x = func_A_pts[0][0]
        # Initial point: evaluate F_A at its start, then pass that to F_B
        y_val_A = func_A_pts[0][1] # eval_func(func_A_pts_tuple, current_x) would be func_A_pts[0][1]
        composed_raw_pts.append((current_x, eval_func(func_B_pts_tuple, y_val_A)))

        while ptr_A < len(func_A_pts) - 1 and ptr_B < len(func_B_pts) - 1:
            x_A0, y_A0 = func_A_pts[ptr_A]
            x_A1, y_A1 = func_A_pts[ptr_A+1]
            
            # Target y-value from F_A's perspective is y_A1 (where F_A's segment ends)
            # Target y-value from F_B's perspective (as a value in F_A's range) is func_B_pts[ptr_B+1][0]
            y_target_from_B_domain = func_B_pts[ptr_B+1][0]

            if y_A0 == y_A1: # Case 1: F_A is flat on current segment [x_A0, x_A1]
                # The composed function F_C value is F_B(y_A0) which is constant.
                # Next x-breakpoint is x_A1.
                current_x = x_A1
                # y_val_A is y_A0 (or y_A1).
                # F_C(x_A1) = F_B(y_A0)
                composed_raw_pts.append((current_x, eval_func(func_B_pts_tuple, y_A0)))
                ptr_A += 1
            # Case 2: F_A is not flat. Compare y_A1 and y_target_from_B_domain.
            # All our functions are non-decreasing, so y_A0 <= y_A1.
            elif y_A1 <= y_target_from_B_domain: 
                # F_A's segment images ends at or before y_target_from_B_domain is reached.
                # So, the next X-breakpoint is x_A1.
                # F_C(x_A1) = F_B(y_A1)
                composed_raw_pts.append((x_A1, eval_func(func_B_pts_tuple, y_A1)))
                if y_A1 == y_target_from_B_domain: # If they align, advance ptr_B as well
                    ptr_B += 1
                ptr_A += 1
            else: # y_A1 > y_target_from_B_domain. (So y_A0 < y_target_from_B_domain < y_A1)
                # F_A's value hits y_target_from_B_domain before reaching y_A1.
                # Preimage x = F_A^{-1}(y_target_from_B_domain) is new X-breakpoint.
                # F_C(x) = F_B(y_target_from_B_domain) = func_B_pts[ptr_B+1][1]
                x_preimage = eval_inverse_func(func_A_pts, ptr_A, y_target_from_B_domain)
                if x_preimage is not None: # If valid integer preimage exists
                     composed_raw_pts.append((x_preimage, func_B_pts[ptr_B+1][1]))
                ptr_B += 1
        
        # If F_A still has points left, F_B's breakpoints are all processed.
        while ptr_A < len(func_A_pts) - 1:
            x_A1, y_A1 = func_A_pts[ptr_A+1]
            composed_raw_pts.append((x_A1, eval_func(func_B_pts_tuple, y_A1)))
            ptr_A += 1
            
        # Sort by x and remove duplicates before cleaning, as merging logic can add points out of order or duplicate x.
        composed_raw_pts.sort() 
        
        final_pts = []
        if composed_raw_pts:
            final_pts.append(composed_raw_pts[0])
            for i in range(1, len(composed_raw_pts)):
                if composed_raw_pts[i][0] > final_pts[-1][0]:
                    final_pts.append(composed_raw_pts[i])
                else: # Same X, update Y to the one from later point (could be F_B breakpoint mapping)
                    final_pts[-1] = composed_raw_pts[i]
        
        return clean_func_pts(final_pts)


    def get_leaf_func(contest_idx):
        L_contest, R_contest = contests[contest_idx]
        
        # Points for h_k(Y) = Y + I(L_k <= Y <= R_k) over domain [MIN_COORD, MAX_COORD]
        pts = []
        
        pts.append((MIN_COORD, MIN_COORD + (1 if L_contest <= MIN_COORD <= R_contest else 0)))
        
        if L_contest > MIN_COORD and L_contest <= MAX_COORD:
            pts.append((L_contest, L_contest + 1))
        
        if R_contest + 1 > MIN_COORD and R_contest + 1 <= MAX_COORD:
             # Value at R_contest is R_contest+1. Value at R_contest+1 is R_contest+1.
            pts.append((R_contest + 1, R_contest + 1)) 
            
        pts.append((MAX_COORD, MAX_COORD + (1 if L_contest <= MAX_COORD <= R_contest else 0)))
        
        pts.sort() # Ensure points are sorted by x-coordinate
        
        # Remove duplicate x-coordinates, keeping the one that implies change if multiple generated for same x
        # (e.g. if MIN_COORD == L_contest). The current logic should be fine due to conditions.
        unique_x_pts = []
        if pts:
            unique_x_pts.append(pts[0])
            for i in range(1, len(pts)):
                if pts[i][0] > unique_x_pts[-1][0]:
                    unique_x_pts.append(pts[i])
                # else: pts[i] has same x as unique_x_pts[-1]. Overwrite with pts[i]
                # because h_k(Y) is well-defined. If L_contest = MIN_COORD, (L_contest, L_contest+1) is the correct one.
                # The specific h_k(L_contest) or h_k(R_contest+1) values are the 'true' values at breakpoints.
                # The current list `pts` has (X, h_k(X)). So if X is repeated, h_k(X) must be same.
                # So it's safe to just take points with distinct X.
        
        return clean_func_pts(unique_x_pts)

    # --- Segment Tree ---
    tree = [None] * (4 * N)

    def build_seg_tree(node_idx, seg_L_contest_idx, seg_R_contest_idx):
        # seg_L/R_contest_idx are indices into the `contests` array
        if seg_L_contest_idx == seg_R_contest_idx:
            tree[node_idx] = get_leaf_func(seg_L_contest_idx)
            return
        
        mid = (seg_L_contest_idx + seg_R_contest_idx) // 2
        build_seg_tree(2 * node_idx, seg_L_contest_idx, mid)
        build_seg_tree(2 * node_idx + 1, mid + 1, seg_R_contest_idx)
        
        func_A_pts = tree[2 * node_idx]     # Corresponds to F_{first_half}
        func_B_pts = tree[2 * node_idx + 1] # Corresponds to F_{second_half}
        # tree[node_idx] stores F_{second_half} composed with F_{first_half}
        tree[node_idx] = compose_funcs(func_B_pts, func_A_pts)

    build_seg_tree(1, 0, N - 1)
    final_func_pts_tuple = tree[1]

    # --- Answer Queries ---
    results = [0] * Q_count
    for x_val, original_idx in queries_input:
        # Clear memoization for eval_func if it grows too large between queries,
        # or if it's not effective. For now, assume it's fine.
        results[original_idx] = eval_func(final_func_pts_tuple, x_val)

    for res in results:
        sys.stdout.write(str(res) + "
")

solve()