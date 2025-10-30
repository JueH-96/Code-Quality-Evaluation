# YOUR CODE HERE
import sys
import itertools

# Function to solve the problem
def solve():
    # Read input N (number of elements) and K (number of elements to choose) from stdin
    N, K = map(int, sys.stdin.readline().split())
    # Read the sequence A of N non-negative integers from stdin
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize the maximum XOR sum found so far. Since all A_i are non-negative,
    # their XOR sum will also be non-negative. Initializing with 0 is correct.
    max_xor_found = 0 

    # The problem asks for the maximum XOR sum among all combinations of K distinct elements from A.
    # It is guaranteed that the number of combinations, C(N, K), is at most 10^6.
    
    # We can solve this by iterating through all combinations. There are two main strategies:
    # 1. Iterate through all combinations of K elements to *include*. Compute XOR sum for each.
    # 2. Iterate through all combinations of N-K elements to *exclude*. Compute XOR sum for each using the total XOR sum property.
    
    # Since C(N, K) = C(N, N-K), the number of combinations is the same for both strategies.
    # However, the work done per combination differs: O(K) for strategy 1 and O(N-K) for strategy 2 (after an O(N) preprocessing step).
    # We choose the strategy that involves fewer elements per combination calculation, i.e., based on min(K, N-K).

    # Check if K is less than or equal to N - K. This is equivalent to K <= N/2.
    if K <= N - K:
        # Strategy 1: Iterate through combinations of K indices to *include*.
        # This strategy is generally more efficient when K is small relative to N.
        
        # Create a range of indices from 0 to N-1.
        indices = range(N)
        # Use itertools.combinations to generate an iterator for all combinations of K indices.
        comb_iter = itertools.combinations(indices, K)
        
        # Variable to store the maximum XOR sum found using this strategy.
        current_max_xor = 0
        
        # Iterate through each combination of K indices.
        for combo_indices in comb_iter:
            # Calculate the XOR sum for the elements A[idx] where idx is in the current combination.
            current_xor = 0
            for idx in combo_indices:
                current_xor ^= A[idx]
            
            # Update the maximum XOR sum found so far if the current combination's XOR sum is greater.
            current_max_xor = max(current_max_xor, current_xor)
        
        # After checking all combinations, the maximum XOR sum found is the answer for this case.
        max_xor_found = current_max_xor

    else: # K > N - K, which is equivalent to K > N/2.
        # Strategy 2: Iterate through combinations of N-K indices to *exclude*.
        # This strategy is generally more efficient when K is large relative to N (meaning N-K is small).
        
        # Calculate the number of elements to exclude.
        k_exclude = N - K
        
        # Calculate the total XOR sum of all elements in the sequence A. This preprocessing takes O(N) time.
        total_xor = 0
        for x in A:
            total_xor ^= x

        # Create a range of indices from 0 to N-1.
        indices = range(N)
        # Use itertools.combinations to generate an iterator for all combinations of N-K indices to exclude.
        comb_iter = itertools.combinations(indices, k_exclude)

        # Variable to store the maximum XOR sum found using this strategy.
        current_max_xor = 0
        
        # Iterate through each combination of N-K indices to exclude.
        for combo_indices_exclude in comb_iter:
            # Calculate the XOR sum of the elements corresponding to the excluded indices.
            xor_excluded = 0
            for idx in combo_indices_exclude:
                xor_excluded ^= A[idx]
            
            # The XOR sum of the *included* elements can be found using the property:
            # (XOR sum of included) = (Total XOR sum) XOR (XOR sum of excluded).
            # This works because for any numbers X, Y: (X XOR Y) XOR Y = X.
            # Let Total XOR = S, XOR of included = P, XOR of excluded = Q. Then S = P XOR Q.
            # Therefore, P = S XOR Q.
            current_xor_included = total_xor ^ xor_excluded
            
            # Update the maximum XOR sum found so far if the current included set's XOR sum is greater.
            current_max_xor = max(current_max_xor, current_xor_included)

        # After checking all combinations, the maximum XOR sum found is the answer for this case.
        max_xor_found = current_max_xor

    # Print the final maximum XOR sum found to standard output.
    print(max_xor_found)

# Execute the solve function when the script is run.
solve()