import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read input sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Find the maximum value M in the sequence A. This determines the size of auxiliary arrays.
    M = 0
    # Since N >= 2, the sequence A is guaranteed to be non-empty.
    for x in A:
        if x > M:
            M = x
    
    # According to constraints, 1 <= A_i <= 10^6, so M >= 1.
    # If M were 0 (e.g., if A_i could be 0), we would need careful handling.
    # But M >= 1 ensures our array indexing and loops are valid.

    # Create a frequency count array `counts` for values from 1 to M.
    # `counts[x]` will store the number of times value x appears in A.
    # Size M+1 to accommodate index M (0-based indexing).
    counts = [0] * (M + 1)
    for x in A:
        counts[x] += 1

    # Create a prefix sum array `prefix_counts`.
    # `prefix_counts[i]` will store the total count of elements in A with values less than or equal to i.
    # This allows efficient querying of counts within a range [L, R].
    prefix_counts = [0] * (M + 1)
    for i in range(1, M + 1):
        prefix_counts[i] = prefix_counts[i-1] + counts[i]

    # Initialize the total sum `total_S`. Python integers handle arbitrary size, so overflow is not an issue.
    total_S = 0

    # Part 1: Calculate contribution from pairs (i, j) where A_i = A_j.
    # For each value v present in A, if it appears `counts[v]` times, there are C(counts[v], 2) pairs (i, j) with i < j such that A_i = A_j = v.
    # The term for such pairs is floor(max(A_i, A_j) / min(A_i, A_j)) = floor(v/v) = floor(1) = 1.
    # So the total contribution is sum over v of C(counts[v], 2).
    for v in range(1, M + 1):
        if counts[v] > 1:
            # Calculate C(counts[v], 2) = counts[v] * (counts[v] - 1) / 2
            # Use integer division // to ensure the result is an integer.
            total_S += counts[v] * (counts[v] - 1) // 2

    # Part 2: Calculate contribution from pairs (i, j) where A_i != A_j.
    # The total contribution is Sum_{1 <= i < j <= N, A_i != A_j} floor(max(A_i, A_j) / min(A_i, A_j)).
    # This can be rewritten by grouping pairs based on their values (v, w):
    # Sum_{v < w} floor(w/v) * (number of pairs (i, j) such that {A_i, A_j} = {v, w}).
    # The number of such pairs is counts[v] * counts[w].
    # Total contribution = Sum_{1 <= v < w <= M} floor(w/v) * counts[v] * counts[w].
    
    # To compute this efficiently, we rearrange the sum:
    # Sum_{v=1 to M} counts[v] * (Sum_{w=v+1 to M} floor(w/v) * counts[w]).
    # Let S_{1,v} = Sum_{w=v+1 to M} floor(w/v) * counts[w].
    # We compute S_{1,v} for each v and add counts[v] * S_{1,v} to total_S.

    # Iterate through each possible value v from 1 to M.
    for v in range(1, M + 1):
        # If value v does not appear in A, it contributes nothing.
        if counts[v] == 0:
            continue 
        
        # Accumulator for the inner sum S_{1,v} related to this v.
        S_1_v = 0 
        
        # Efficiently compute S_{1,v} by iterating through w > v.
        # We group consecutive w values that have the same floor quotient k = floor(w/v).
        # Start checking from the first possible w, which is v+1.
        curr_w = v + 1
        while curr_w <= M:
            
            # Determine the quotient k = floor(w/v) for the current starting w `curr_w`.
            # Since curr_w >= v+1 and v >= 1, k = curr_w // v will be at least 1.
            k = curr_w // v 
            
            # Determine the end of the range of w values that yield the same quotient k.
            # The range for a fixed k is mathematically [kv, (k+1)v - 1].
            # The segment of w values we process in this step starts at `curr_w`.
            # The maximum w in this k-block is min(M, (k+1)*v - 1).
            range_end_w = min(M, (k + 1) * v - 1)
            
            # Calculate the number of elements in A whose values fall in the segment [curr_w, range_end_w].
            # Use prefix sums for O(1) query: count = prefix_counts[range_end_w] - prefix_counts[curr_w - 1].
            count_in_segment = prefix_counts[range_end_w] - prefix_counts[curr_w - 1]
            
            # If there are elements in this segment (count > 0), add their contribution to S_1_v.
            # Each element w in this segment contributes floor(w/v) = k to the sum.
            # Total contribution from this segment is k * count_in_segment.
            if count_in_segment > 0:
                S_1_v += k * count_in_segment
            
            # Move to the start of the next segment. The next w to check is range_end_w + 1.
            curr_w = range_end_w + 1
            # If range_end_w was M, then curr_w becomes M+1, and the loop condition `curr_w <= M` will fail, correctly terminating the loop for v.

        # Add the total contribution associated with value v to the overall sum.
        # Multiply S_1_v by counts[v] because each of the counts[v] instances of v forms pairs with elements w > v.
        total_S += counts[v] * S_1_v

    # Print the final computed total sum.
    print(total_S)

# Execute the solve function to run the program logic.
solve()