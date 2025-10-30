import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    if N == 1:
        print(1)
        return

    # total_count will store the number of arithmetic progressions.
    # All sequences of length 1 are APs. There are N such APs.
    total_count = N
    
    i = 0
    # This loop processes segments of constant differences.
    # i is the starting index (0-indexed) of an element A[i].
    # We consider APs involving A[i], A[i+1], ...
    # So, i must be less than N-1 to have at least A[i] and A[i+1].
    while i < N - 1:
        current_diff = A[i+1] - A[i]
        
        # run_length counts how many consecutive pairs (A[x], A[x+1])
        # have this same current_diff, starting from (A[i], A[i+1]).
        run_length = 0
        
        # k scans from i. For each k, we check the difference A[k+1]-A[k].
        k = i 
        while k < N - 1: # k ensures A[k+1] is a valid index
            if A[k+1] - A[k] == current_diff:
                run_length += 1
                k += 1 # Move to check the next pair
            else:
                # Difference changed, this run of current_diff ends.
                break
        
        # A run of `run_length` identical differences `current_diff`
        # corresponds to an AP segment of `run_length + 1` elements in A.
        # These elements are A[i], A[i+1], ..., A[i+run_length].
        # Let L_elements = run_length + 1.
        # This AP segment contains L_elements * (L_elements - 1) / 2 subsegments
        # that are APs of length >= 2.
        # This is equivalent to (run_length + 1) * run_length / 2.
        
        # Add this count to total_count.
        # Since total_count was initialized to N (counting all length-1 APs),
        # we are adding counts for APs of length >= 2.
        count_len_ge_2_aps_in_segment = (run_length * (run_length + 1)) // 2 - run_length
        # The term run_length * (run_length + 1) // 2 is sum of integers 1 to run_length.
        # If run_length = 1 (e.g. AP (x,y)), sum = 1. (x,y) itself.
        # If run_length = 2 (e.g. AP (x,y,z)), sum = 1+2 = 3. (x,y), (y,z), (x,y,z).
        # These are APs of length >= 2.
        # So, the correct number of APs of length >= 2 derived from this run is sum(1...run_length).
        # This is run_length * (run_length + 1) // 2.

        # The derivation L_elements * (L_elements - 1) // 2 is correct.
        # L_elements = run_length + 1.
        # So, (run_length + 1) * ((run_length + 1) - 1) // 2
        #  = (run_length + 1) * run_length // 2.
        
        total_count += (run_length * (run_length + 1)) // 2 - run_length
        
        # Advance i. The next iteration will start checking from index k.
        # The pairs from A[i]...A[k-1] (i.e., diffs diffs[i]...diffs[k-1]) have been processed.
        i = k 
    
    print(total_count)

solve()