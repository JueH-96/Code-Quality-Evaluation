import sys
import itertools

def solve():
    N, K = map(int, sys.stdin.readline().split())
    
    functions = []
    for i in range(N):
        A, B = map(int, sys.stdin.readline().split())
        functions.append((A, B, i)) # Store (A, B, original_index) to preserve uniqueness

    # Sort functions by A in descending order, then by B in descending order.
    # This heuristic aims to prioritize functions that contribute most to the product of A's,
    # and then to the B terms when A's are equal.
    functions.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Determine the number of candidates to consider for permutations.
    # For K=10, P(12,10) = 12!/2! = 2.395 * 10^8 permutations.
    # This number of permutations, each involving K arithmetic operations,
    # is generally at the upper limit for typical competitive programming time limits (1-2 seconds) in Python.
    # However, itertools.permutations is implemented in C, and Python handles large integers efficiently,
    # making this approach potentially viable.
    num_candidates = min(N, 12) 
    
    candidate_functions = functions[:num_candidates]

    max_val = 0 # Initialize max_val to 0, since A_i, B_i >= 1 and starting x=1, values are always positive.

    # Iterate through all permutations of K functions chosen from candidate_functions.
    # `p_indices` is a tuple of indices into `candidate_functions`.
    # Each permutation represents an ordered sequence of K distinct functions (p_1, p_2, ..., p_K).
    for p_indices in itertools.permutations(range(len(candidate_functions)), K):
        current_value = 1 # Initial value for the innermost function f_pK(1)
        
        # Apply functions from right to left (innermost to outermost)
        # The permutation (p_1, p_2, ..., p_K) implies f_{p_1}(f_{p_2}(...f_{p_K}(1)...)).
        # So, we first apply f_{p_K} to 1, then f_{p_{K-1}} to the result, and so on,
        # until f_{p_1} is applied.
        
        # We iterate through the permutation `p_indices` in reverse order.
        # p_indices[K-1] corresponds to the index of f_pK (innermost function)
        # p_indices[0] corresponds to the index of f_p1 (outermost function)
        for j in range(K - 1, -1, -1):
            A, B, _ = candidate_functions[p_indices[j]]
            current_value = A * current_value + B
        
        # Update the overall maximum value found
        max_val = max(max_val, current_value)

    print(max_val)

# Call the solve function to run the program
solve()