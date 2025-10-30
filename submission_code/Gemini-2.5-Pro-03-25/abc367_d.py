# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the list of steps A_i as integers. A[i] in the list corresponds to A_{i+1} in the problem statement (1-based indexing).
    A = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums P_i. P_i represents the clockwise distance from rest area 1 to rest area i.
    # We use 1-based indexing for the P array to align with the problem statement (indices 1 to N).
    # P[0] is unused. The array size is 2*N + 1 to accommodate indices up to 2N for the extended sequence.
    P = [0] * (2 * N + 1) 
    
    # P[1] = 0 by definition (distance from rest area 1 to itself).
    
    current_sum = 0
    # Calculate P[2] through P[N].
    # P[i] = sum_{k=1}^{i-1} A_k (using 1-based indexing for A as in problem statement).
    # This translates to sum_{k=0}^{i-2} A[k] using 0-based indexing for the list A.
    for i in range(N - 1): # Python loop index `i` goes from 0 to N-2
        current_sum += A[i] # A[i] are the steps from rest area i+1 to i+2.
        P[i+2] = current_sum # Store the cumulative sum. P[2] = A[0], P[3] = A[0]+A[1], ..., P[N] = A[0]+..+A[N-2].
    
    # Calculate the total circumference C = sum_{k=1}^{N} A_k (1-based A).
    # This corresponds to sum_{k=0}^{N-1} A[k] (0-based A list).
    # After the loop, current_sum holds sum_{k=0}^{N-2} A[k]. We need to add the last step A[N-1] (which is A_N in 1-based).
    C = current_sum + A[N-1]

    # Calculate P_i for indices i = N+1 to 2N using the property P_{N+i} = C + P_i.
    # This creates an extended sequence of prefix sums that represents distances for paths wrapping around the lake.
    for i in range(1, N + 1):
        P[N+i] = C + P[i]

    # Compute p_i = P_i mod M for all relevant indices i = 1 to 2N.
    # p_i stores the prefix sum modulo M. This is the key property we need.
    p = [0] * (2 * N + 1)
    for i in range(1, 2 * N + 1):
        p[i] = P[i] % M

    # Group indices based on their p_i value (prefix sum modulo M).
    # Use a defaultdict(list) for efficient grouping. Keys will be the modulo values (0 to M-1),
    # values will be lists containing the indices i where p[i] equals the key.
    indices_by_value = defaultdict(list)
    for i in range(1, 2 * N + 1):
        indices_by_value[p[i]].append(i) # Append index i to the list associated with value p[i].

    total_pairs = 0 # Initialize the total count of valid pairs (s, t).

    # Iterate through each distinct modulo value v that appeared in the p array.
    for v in indices_by_value:
        indices = indices_by_value[v] # Get the list of indices i where p[i] = v. Indices are naturally sorted as they were added sequentially.
        m = len(indices) # The number of times this value v appeared in the p sequence.
        
        # If a value appears less than twice, it cannot form any pair (s, k) such that p_s = p_k and s < k.
        if m < 2: 
            continue

        # Apply the two-pointer technique to efficiently count pairs (s, k) satisfying the conditions.
        p1 = 0 # Left pointer, iterates through potential start indices s from the `indices` list.
        p2 = 0 # Right pointer, used to find the upper bound for valid end indices k.
        
        current_v_pairs = 0 # Counter for pairs found specifically for this value v.
        
        # Iterate through all possible start indices s = indices[p1] for the current value v.
        for p1 in range(m):
            i = indices[p1] # This `i` is the actual index s (from 1 to 2N). It represents a potential starting rest area.
            
            # The problem requires the starting rest area 's' to be one of the original N areas, i.e., 1 <= s <= N.
            if i > N: 
                # Since the `indices` list is sorted, all subsequent indices `indices[p1+1], ...` will also be greater than N.
                # No more valid start indices 's' can be found for this value v. We can break the loop.
                break 

            # The destination node 't' must be reachable from 's' clockwise within one full loop around the lake.
            # This means the distance D(s, t) should be the minimum clockwise distance.
            # In terms of the extended prefix sum indices, this translates to k <= s + N - 1.
            target_upper_bound = i + N - 1
            
            # Ensure the right pointer p2 starts ahead of p1, since we need pairs (s, k) where s < k.
            # These correspond to pairs of indices (p1, p2_idx) from the `indices` list where p1 < p2_idx.
            if p2 <= p1:
                p2 = p1 + 1
            
            # Advance the right pointer p2 as long as the corresponding index `indices[p2]` is within the valid range for k.
            # `indices[p2]` represents a potential index k. The condition is k <= target_upper_bound.
            # The loop moves p2 to the first position where indices[p2] > target_upper_bound (or until p2 reaches the end of the list).
            while p2 < m and indices[p2] <= target_upper_bound:
                 p2 += 1

            # After the while loop, p2 points to the first index in `indices` list that is strictly greater than target_upper_bound (or p2 == m).
            # The indices in the `indices` list that satisfy the conditions (being > p1 and corresponding k <= target_upper_bound) are:
            # at positions p1+1, p1+2, ..., p2-1.
            # The number of such valid indices k for the current start index s = indices[p1] is (p2 - 1) - (p1 + 1) + 1 = p2 - p1 - 1.
            # Simply, it's the count of elements between index p1 (exclusive) and p2 (exclusive).
            count_for_p1 = p2 - (p1 + 1)
            
            # If count_for_p1 is positive, it means we found valid k's for this s=i. Add this count.
            if count_for_p1 > 0:
                 current_v_pairs += count_for_p1
        
        # Add the total count of pairs found for the current value v to the overall total count.
        total_pairs += current_v_pairs

    # Print the final computed total number of pairs (s, t) satisfying the condition.
    print(total_pairs)

# Execute the solve function to run the main logic of the program.
solve()