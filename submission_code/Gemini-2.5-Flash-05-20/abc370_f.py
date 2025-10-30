import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    A_double = A + A
    
    # P_double[i] stores the sum of A_double[0] through A_double[i-1]
    P_double = [0] * (2 * N + 1)
    for i in range(2 * N):
        P_double[i+1] = P_double[i] + A_double[i]

    # Max possible sum for a piece is sum(A), min is 1.
    # Binary search range for the minimum mass X
    low_X = max(A) # Minimum possible X is the max single piece mass (must take at least one piece)
    high_X = sum(A) # Maximum possible X is the total sum (if K=1)
    
    ans_X = 0

    # MAX_LOG_K is ceil(log2(2*N)) to cover all possible segment lengths up to 2*N
    MAX_LOG_K = (2 * N).bit_length() 

    # jump[p][i] stores the starting index of the segment 2^p steps after starting at i.
    # If starting at index i and taking 2^p segments, the next segment begins at jump[p][i].
    jump = [[-1] * (2 * N) for _ in range(MAX_LOG_K)]

    # Function to check if a minimum mass 'm' is achievable for all K people.
    # It also populates the 'jump' table for the given 'm'.
    def check(m):
        # 1. Compute next_start_idx[i] for i from 0 to 2*N-1.
        #    next_start_idx[i] is the earliest index j such that sum(A_double[i...j-1]) >= m.
        #    This 'j' will be the starting index of the next segment.
        
        # Using two-pointers for O(N) calculation of next_start_idx
        R = 0
        current_sum = 0
        temp_next_start_idx = [-1] * (2 * N)
        for L in range(2 * N):
            # Extend window to the right until current_sum >= m or R reaches end
            while R < 2 * N and current_sum < m:
                current_sum += A_double[R]
                R += 1
            
            if current_sum >= m:
                temp_next_start_idx[L] = R
            else: # Cannot form a segment of mass m even by taking all remaining pieces
                temp_next_start_idx[L] = -1 
                
            # Shrink window from the left for the next iteration
            current_sum -= A_double[L]

        # 2. Populate jump[0] table
        for i in range(2 * N):
            jump[0][i] = temp_next_start_idx[i]
        
        # 3. Build the binary lifting jump table
        for p in range(1, MAX_LOG_K):
            for i in range(2 * N):
                if jump[p-1][i] != -1:
                    jump[p][i] = jump[p-1][jump[p-1][i]]
                else:
                    jump[p][i] = -1 # Cannot make 2^p segments if 2^(p-1) segments already failed

        # 4. For each possible starting position in the original array (0 to N-1),
        #    check if K segments can be formed covering exactly N pieces.
        for start_idx in range(N):
            current_pos_sim = start_idx # Position in A_double
            num_segments_sim = 0
            
            for p in range(MAX_LOG_K - 1, -1, -1): # Iterate from largest power of 2 down to 0
                if (K >> p) & 1: # If the p-th bit of K is set, means we need to take 2^p segments
                    # Check if the jump is possible and within the N-piece boundary for circular array
                    if current_pos_sim == -1 or current_pos_sim >= start_idx + N:
                        num_segments_sim = K + 1 # Mark as impossible: jump failed or exceeded N pieces
                        break
                    current_pos_sim = jump[p][current_pos_sim]
                    num_segments_sim += (1 << p)

            # Check if exactly K segments were formed and they cover exactly N pieces.
            if num_segments_sim == K and current_pos_sim == start_idx + N:
                return True
        return False

    # Binary search for the maximum X
    while low_X <= high_X:
        mid_X = (low_X + high_X) // 2
        if mid_X == 0: # Avoid infinite loop if A_i can be 0 or small, though constraint says A_i >= 1
            low_X = 1
            continue
        
        if check(mid_X):
            ans_X = mid_X
            low_X = mid_X + 1
        else:
            high_X = mid_X - 1

    # Now find the number of never-cut lines (Y) for ans_X
    # Re-run check(ans_X) to ensure the 'jump' table is populated with values specific to ans_X.
    # This is important because 'jump' is a global or heavily-scoped variable modified by 'check'.
    check(ans_X) 

    all_cut_lines_set = set() # Stores indices of cut lines that are cut in *any* optimal division

    for start_idx in range(N):
        current_pos_trace = start_idx # Position in A_double
        num_segments_trace = 0
        
        # Similar binary lifting to verify if start_idx yields an optimal division
        # This part ensures we only consider divisions that achieve ans_X
        # Note: K jumps are needed to cover all K segments. K-1 explicit internal cuts.
        # The K-th cut wraps around to start_idx.
        
        target_pos_after_K_segments = start_idx
        # Calculate where we would end up after K segments
        for p in range(MAX_LOG_K - 1, -1, -1):
            if (K >> p) & 1:
                if target_pos_after_K_segments == -1 or target_pos_after_K_segments >= start_idx + N:
                    target_pos_after_K_segments = -1 # Mark as impossible
                    break
                target_pos_after_K_segments = jump[p][target_pos_after_K_segments]
        
        # If this starting point results in a valid K-segment partition covering N pieces
        if target_pos_after_K_segments != -1 and target_pos_after_K_segments == start_idx + N:
            # This 'start_idx' leads to a valid optimal division.
            # Collect the actual cut lines made by this division.
            current_trace_pos = start_idx
            # There are K segments. This means K cuts. One cut is implicit by circularity.
            # We explicitly list the K-1 cuts that subdivide the N pieces.
            for _ in range(K - 1): 
                next_segment_start_idx = jump[0][current_trace_pos]
                
                # If next_segment_start_idx is invalid or suggests exceeding N pieces, break
                if next_segment_start_idx == -1 or next_segment_start_idx >= start_idx + N:
                    break 
                
                # A cut is made after piece (next_segment_start_idx - 1) % N.
                # Cut line 'i' is between piece 'i' and 'i+1'. So after piece 'i', it's cut line 'i'.
                # (using 0-indexed pieces, 0-indexed cut lines)
                all_cut_lines_set.add((next_segment_start_idx - 1) % N)
                current_trace_pos = next_segment_start_idx
        
    ans_Y = N - len(all_cut_lines_set)

    sys.stdout.write(f"{ans_X} {ans_Y}
")

solve()