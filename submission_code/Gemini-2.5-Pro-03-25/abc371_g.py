# YOUR CODE HERE
import sys

# Function to find the lexicographically smallest cyclic shift of a list using Duval Algorithm variant
# Based on https://cp-algorithms.com/string/lyndon_factorization.html minimal cyclic shift section.
# The algorithm computes the Lyndon factorization. The minimum rotation starts at the beginning of the minimal Lyndon factor found during factorization.
# Returns the shift amount (k) which is the starting index of the minimal rotation in the original array, 
# and the minimum shifted list itself.
def find_min_cyclic_shift_info(arr):
    S = arr
    N = len(S)
    if N == 0:
        return 0, []
    
    i = 0 # Start index of the current block being processed
    min_shift_idx = 0 # Start index of the minimal rotation found so far
    
    # The outer loop iterates through the blocks of identical Lyndon words that form the factorization of S.
    while i < N:
        # We store the start index `i` of the current block as the best candidate found so far.
        # The algorithm ensures that the `i` that leads to the smallest Lyndon word comparison becomes the final `min_shift_idx`.
        min_shift_idx = i 
        
        # j scans forward to find the end of the block
        j = i + 1 
        # k tracks the index within the initial word of the block for comparison S[k] vs S[j]
        k = i 
        
        # Inner loop extends the comparison as long as characters match or S[j] > S[k]
        while j < N and S[k] <= S[j]:
            if S[k] < S[j]:
                # If S[j] is strictly greater, the potential Lyndon word starting at i is smaller than what starts at j.
                # Reset k to the beginning of the word (index i), indicating the start of a new potential Lyndon word comparison.
                k = i
            else: # S[k] == S[j]
                # Characters match, advance k to compare the next character of the word.
                k += 1
            # Always advance j to check the next character of S.
            j += 1
            
        # After the inner loop, S[i...j-1] consists of repetitions of a Lyndon word.
        # The length of this Lyndon word is l = j - k.
        
        # Move `i` past all identical Lyndon words found in this block.
        # This ensures that the next iteration starts at the beginning of a new, potentially different Lyndon word.
        while i <= k: 
             # Advance `i` by the length of the Lyndon word found.
             # `k` points to `i + offset within word` where match ended, so `j-k` is the word length.
            i += (j - k)

    # `min_shift_idx` now holds the starting index of the lexicographically smallest cyclic shift.
    
    # Reconstruct the minimum sequence based on the starting index `min_shift_idx`
    # The minimal sequence is arr rotated left by min_shift_idx positions.
    min_seq = S[min_shift_idx:] + S[:min_shift_idx]
    
    # Return the shift amount (which is `min_shift_idx`) and the sequence
    return min_shift_idx, min_seq


def solve():
    N = int(sys.stdin.readline())
    # Read P and A as strings and convert to lists of integers
    P_str = sys.stdin.readline().split()
    A_str = sys.stdin.readline().split()
    # Adjust P to 0-based indexing for array access
    P = [int(p) - 1 for p in P_str] 
    # A values represent numbers 1..N, keep them as integers.
    A = [int(a) for a in A_str] 

    visited = [False] * N
    cycles = []

    # Find cycles of permutation P using DFS
    for i in range(N):
        if not visited[i]:
            current_cycle_indices = []
            curr = i
            # Follow the cycle until we encounter a visited node (which must be the start of the cycle)
            while not visited[curr]:
                visited[curr] = True
                current_cycle_indices.append(curr)
                curr = P[curr]
            # Append the found cycle indices list if it's not empty
            if current_cycle_indices: 
               cycles.append(current_cycle_indices)

    # B will store the resulting permutation corresponding to the minimal configuration
    # Initialize B with zeros or any placeholder value.
    B = [0] * N 
    
    # Process each cycle found in P
    for cycle_indices in cycles:
        L = len(cycle_indices)
        # Skip empty cycles (should not happen for N >= 1 and valid permutation P)
        if L == 0: continue 

        # Get the sequence of values from A corresponding to the indices in the current cycle
        cycle_values = [A[idx] for idx in cycle_indices]
        
        # Find the minimum cyclic shift information using the efficient O(L) algorithm
        # `_` stores the shift amount k, `final_cycle_values` is the lexicographically minimum sequence.
        _ , final_cycle_values = find_min_cyclic_shift_info(cycle_values)
        
        # Place the values from the minimum shifted sequence into the result array B
        # The value at index `idx_in_cycle` of `final_cycle_values` corresponds to the value
        # that should be placed at index `cycle_indices[idx_in_cycle]` in the final permutation B.
        for idx_in_cycle, original_idx in enumerate(cycle_indices):
            B[original_idx] = final_cycle_values[idx_in_cycle]
        
    # Print the resulting permutation B, with elements separated by spaces.
    print(*(B))

# Execute the solve function to run the program.
solve()