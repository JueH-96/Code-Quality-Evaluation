import sys
import bisect

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Precompute prefix sums for '1's, '2's, and '/' characters.
    # pref_ones[i] = count of '1's in S[0...i-1]
    # pref_twos[i] = count of '2's in S[0...i-1]
    # pref_slashes[i] = count of '/' in S[0...i-1]
    # All are 1-indexed for length (e.g., pref_ones[k] is count for S[0...k-1])
    pref_ones = [0] * (N + 1)
    pref_twos = [0] * (N + 1)
    pref_slashes = [0] * (N + 1) 
    
    for i in range(N):
        pref_ones[i+1] = pref_ones[i] + (1 if S[i] == '1' else 0)
        pref_twos[i+1] = pref_twos[i] + (1 if S[i] == '2' else 0)
        pref_slashes[i+1] = pref_slashes[i] + (1 if S[i] == '/' else 0) 

    # Segment Tree structure:
    # Each node 'v' in the segment tree (representing a range of original string indices [start, end])
    # stores a tuple: (points_list, suffix_min_p2_list)
    # `points_list`: A list of (pref_ones[k], pref_twos[k+1]) tuples for all k in [start, end] where S[k] == '/'.
    #                This list is sorted by pref_ones[k] values.
    # `suffix_min_p2_list`: A parallel list where suffix_min_p2_list[i] stores the minimum
    #                       of pref_twos[k+1] values for points from index i to the end of `points_list`.
    tree = [None] * (4 * N) # Array to store segment tree nodes. Max size is 4*N.

    # Build function for the segment tree
    # Populates `tree` array recursively.
    # node: current node index in the `tree` array
    # start, end: range of indices in the original string S covered by this node
    def build(node, start, end):
        if start == end: # Leaf node
            if S[start] == '/':
                # A single point (pref_ones[start], pref_twos[start+1])
                tree[node] = ([(pref_ones[start], pref_twos[start+1])], [pref_twos[start+1]])
            else:
                tree[node] = ([], []) # If S[start] is not '/', no point in this leaf
        else: # Internal node
            mid = (start + end) // 2
            build(2 * node, start, mid) # Build left child
            build(2 * node + 1, mid + 1, end) # Build right child
            
            # Merge children's points lists
            left_points, _ = tree[2 * node]
            right_points, _ = tree[2 * node + 1]
            
            merged_points = []
            # Merge two sorted lists into one sorted list
            p_left, p_right = 0, 0
            while p_left < len(left_points) and p_right < len(right_points):
                if left_points[p_left][0] <= right_points[p_right][0]:
                    merged_points.append(left_points[p_left])
                    p_left += 1
                else:
                    merged_points.append(right_points[p_right])
                    p_right += 1
            merged_points.extend(left_points[p_left:]) # Add remaining elements from left
            merged_points.extend(right_points[p_right:]) # Add remaining elements from right
            
            # Calculate suffix minimums for the merged_points list
            suffix_min_p2 = [0] * len(merged_points)
            if merged_points: # Ensure list is not empty before accessing elements
                suffix_min_p2[len(merged_points) - 1] = merged_points[len(merged_points) - 1][1]
                for i in range(len(merged_points) - 2, -1, -1):
                    suffix_min_p2[i] = min(suffix_min_p2[i+1], merged_points[i][1])
            
            tree[node] = (merged_points, suffix_min_p2)

    # Query function for the segment tree
    # Checks if there exists an index k in [query_L_idx, query_R_idx] (0-indexed)
    # such that S[k] == '/' AND pref_ones[k] >= min_p1_val AND pref_twos[k+1] <= max_p2_val
    def query_seg_tree(node, start, end, query_L_idx, query_R_idx, min_p1_val, max_p2_val):
        # query_L_idx, query_R_idx are the 0-indexed range of 'k' to search within S.
        # node range [start, end] is the range covered by the current segment tree node.
        
        # Case 1: Current node's range is outside the query range
        if query_R_idx < start or end < query_L_idx: 
            return False
        
        # Case 2: No '/' characters in this segment (checked during build)
        if tree[node] == ([], []): 
            return False

        # Case 3: Current node's range is completely within the query range
        if query_L_idx <= start and end <= query_R_idx: 
            points, suffix_min_p2 = tree[node]
            
            # Use bisect_left to find the first point where pref_ones[k] >= min_p1_val
            # The dummy value -1 for p2_val works because actual pref_twos_val are non-negative.
            # bisect_left will return the index where (min_p1_val, -1) could be inserted,
            # maintaining sort order by first element, then second.
            idx = bisect.bisect_left(points, (min_p1_val, -1)) 
            
            if idx < len(points):
                # If such a point is found, check if its corresponding pref_twos[k+1]
                # (or any subsequent pref_twos[k+1] in the suffix list, which suffix_min_p2 handles)
                # is <= max_p2_val.
                if suffix_min_p2[idx] <= max_p2_val:
                    return True
            return False
        
        # Case 4: Partial overlap, recurse on children
        mid = (start + end) // 2
        # Check left child. If found, return True immediately (short-circuit).
        found_in_left = query_seg_tree(2 * node, start, mid, query_L_idx, query_R_idx, min_p1_val, max_p2_val)
        if found_in_left: return True 
        # Check right child.
        found_in_right = query_seg_tree(2 * node + 1, mid + 1, end, query_L_idx, query_R_idx, min_p1_val, max_p2_val)
        return found_in_right

    # This function is used by the binary search for 'm'.
    # It checks if an 11/22 string of length 2*m_val+1 can be formed.
    # It assumes m_val >= 1, as m=0 is handled separately (it just needs a '/').
    def check_m_possible(m_val, query_L_idx, query_R_idx, current_pref_ones_L_minus_1, current_pref_twos_R):
        # Calculate the required minimum number of '1's to the left of '/'
        # and the maximum allowed number of '2's to the right of '/'.
        min_ones_needed = m_val + current_pref_ones_L_minus_1
        max_twos_allowed = current_pref_twos_R - m_val
        
        # Optimization: If the required '1's exceed total '1's in S
        # or if the allowed '2's count becomes negative, it's impossible.
        if min_ones_needed > pref_ones[N] or max_twos_allowed < 0:
            return False
        
        # Perform the segment tree query to find if a suitable '/' exists.
        return query_seg_tree(1, 0, N - 1, query_L_idx, query_R_idx, min_ones_needed, max_twos_allowed)

    # Build the segment tree once for all queries
    build(1, 0, N - 1)

    results = [] # To store answers for all queries
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())

        # Initial check: If no '/' character exists in the substring S[L-1:R],
        # then no 11/22 string can be formed (as it must contain a '/').
        # In this case, the answer is 0.
        if pref_slashes[R] - pref_slashes[L-1] == 0:
            results.append(0)
            continue
            
        # Binary search for the maximum possible 'm' value.
        # 'm' ranges from 0 up to roughly half the length of the substring.
        low = 0
        high = (R - L + 1) // 2 
        
        ans_m = 0 # Stores the maximum 'm' found to be possible.
                  # Initialized to 0, since we already know a '/' exists (length 1 string is possible).
        
        while low <= high:
            mid_m = (low + high) // 2
            
            # Case for m=0: A single '/' is an 11/22 string of length 1 (m=0).
            # We already confirmed a '/' exists for this query if we reach this point.
            # So, m=0 is always possible here. Update ans_m and try larger 'm'.
            if mid_m == 0: 
                ans_m = max(ans_m, mid_m)
                low = mid_m + 1
                continue
            
            # For m >= 1, use the `check_m_possible` function which performs segment tree queries.
            # Pass relevant prefix sums for the current query range.
            if check_m_possible(mid_m, L - 1, R - 1, pref_ones[L-1], pref_twos[R]):
                ans_m = mid_m    # mid_m is possible, try for a larger 'm'
                low = mid_m + 1
            else:
                high = mid_m - 1 # mid_m is not possible, try for a smaller 'm'
        
        # The maximum length of the 11/22 string is 2*m + 1.
        results.append(2 * ans_m + 1)

    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

solve()