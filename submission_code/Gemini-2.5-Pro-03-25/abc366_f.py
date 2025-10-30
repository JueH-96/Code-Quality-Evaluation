# YOUR CODE HERE
import sys
import itertools
from functools import cmp_to_key

def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())
    
    # Read all N functions and store them along with their original index (0 to N-1)
    all_funcs_list = []
    for i in range(N):
        A, B = map(int, sys.stdin.readline().split())
        # Store function data as a dictionary for easy access to A, B, and original id
        all_funcs_list.append({'A': A, 'B': B, 'id': i})

    # --- Candidate Selection Phase ---
    # The strategy is to select a pool of candidate functions that are likely 
    # to be part of the optimal sequence. We select the top K functions based on
    # three criteria: highest A, highest B, and highest A+B (value at x=1).
    # The union of these top K lists forms our candidate pool.
    
    # Sort by A descending and take top K. Use sorted() to create a new list.
    sorted_by_A = sorted(all_funcs_list, key=lambda f: f['A'], reverse=True)
    candidates_A = sorted_by_A[:K]
    
    # Sort by B descending and take top K
    sorted_by_B = sorted(all_funcs_list, key=lambda f: f['B'], reverse=True)
    candidates_B = sorted_by_B[:K]

    # Sort by A+B descending and take top K
    sorted_by_AB = sorted(all_funcs_list, key=lambda f: f['A'] + f['B'], reverse=True)
    candidates_AB = sorted_by_AB[:K]

    # Collect unique candidate function IDs using a set to handle overlaps
    candidate_ids = set()
    for f in candidates_A:
        candidate_ids.add(f['id'])
    for f in candidates_B:
        candidate_ids.add(f['id'])
    for f in candidates_AB:
         candidate_ids.add(f['id'])

    # Create a map from function ID to function data dictionary for efficient lookup
    funcs_map_by_id = {f['id']: f for f in all_funcs_list}

    # Convert the set of candidate IDs to a list. This list contains the IDs
    # of all functions that are potentially part of the optimal sequence.
    candidate_ids_list = list(candidate_ids)
    
    # Initialize the maximum value found so far. 
    # Since all A_i, B_i >= 1, any composition result F(1) will be >= 2.
    # Using -1 ensures any valid result will be larger.
    best_val = -1 

    # --- Sorting Logic Definition ---
    # Define the comparison function for sorting a chosen set of K functions.
    # This function determines the optimal relative order of two functions f1, f2
    # in the sequence p = (p1, ..., pK) based on the derived rule.
    # The rule aims to maximize F(1) = f_p1(...f_pK(1)...).
    # Rule: Functions with A > 1 come first, sorted by increasing B/(A-1).
    #       Functions with A = 1 come last. Order among A=1 functions doesn't matter.
    def compare_funcs(f1, f2):
        # Check if A coefficient is 1 for either function
        A1_is_1 = (f1['A'] == 1)
        A2_is_1 = (f2['A'] == 1)

        if A1_is_1 and A2_is_1:
            # If both A=1, their relative order doesn't impact the final value.
            # Return comparison based on ID for a stable sort order.
            return f1['id'] - f2['id'] 
        elif A1_is_1:
            # f1 has A=1, f2 has A>1. f1 should appear later in the sequence p than f2.
            # According to Python's sort convention, returning positive means f1 comes after f2.
            return 1  
        elif A2_is_1:
            # f1 has A>1, f2 has A=1. f1 should appear earlier in the sequence p than f2.
            # Returning negative means f1 comes before f2.
            return -1 
        else:
            # Both A > 1. Compare B/(A-1) values. We want ascending order.
            # To avoid floating point precision issues, use cross multiplication:
            # B1/(A1-1) < B2/(A2-1)  <=>  B1*(A2-1) < B2*(A1-1)
            val1 = f1['B'] * (f2['A'] - 1)
            val2 = f2['B'] * (f1['A'] - 1)
            
            if val1 < val2:
                return -1 # f1 comes before f2 because its B/(A-1) value is smaller
            elif val1 > val2:
                return 1  # f1 comes after f2 because its B/(A-1) value is larger
            else:
                # If B/(A-1) values are equal, break ties using original ID for stability.
                return f1['id'] - f2['id'] 
    
    # --- Combination Generation and Evaluation Phase ---
    
    # Defensive check: Ensure we have at least K candidates. 
    # This should always be true since N >= K based on problem constraints.
    if len(candidate_ids_list) < K:
         # If N < K this path could be reached, but constraints say N >= K.
         # If N >= K, len(candidate_ids_list) must be >= K because candidates_A alone has K items.
         # The combination generation below would yield nothing if len < K.
         pass # Assume len >= K.

    # Generate all combinations of K distinct function indices from the candidate pool
    # `itertools.combinations` provides an iterator over tuples of indices.
    for combo_indices in itertools.combinations(candidate_ids_list, K):
        
        # Retrieve the full function data dictionaries for the current combination of K indices
        current_combo_funcs = [funcs_map_by_id[idx] for idx in combo_indices]
        
        # Sort this combination of K functions using the defined comparison rule.
        # `cmp_to_key` converts the old-style comparison function to a key function.
        sorted_combo = sorted(current_combo_funcs, key=cmp_to_key(compare_funcs))
        
        # Evaluate the composition F(1) = f_p1(f_p2(...f_pK(1)...))
        # The `sorted_combo` list represents the optimal sequence p = (p1, ..., pK)
        # for this set of K functions, where p1 is sorted_combo[0], etc.
        
        current_val = 1 # Start with the initial input value x=1
        
        # Apply the functions in the sequence composition order:
        # The innermost function f_pK is applied first, then f_{p_{K-1}}, ..., finally f_p1.
        # This means iterating through `sorted_combo` list from the end (index K-1) to the start (index 0).
        for k in range(K - 1, -1, -1):
             # Get the function f_{p_{k+1}} which is at index k in the sorted list
             func = sorted_combo[k] 
             # Apply the function: val = A * val + B
             current_val = func['A'] * current_val + func['B']
        
        # Update the overall maximum value found across all combinations
        # Using max() is concise and clear.
        best_val = max(best_val, current_val)

    # Print the final maximum value found. It must be an integer.
    print(best_val)

# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()