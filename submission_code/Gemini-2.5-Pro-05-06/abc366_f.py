import math
from functools import cmp_to_key
from itertools import combinations

def solve():
    N, K = map(int, input().split())
    functions_orig = []
    for i in range(N):
        A, B = map(int, input().split())
        functions_orig.append({'A': A, 'B': B, 'idx': i})

    # Use a copy for sorting, so functions_orig remains untouched if needed later.
    functions_temp = [{'A': f['A'], 'B': f['B'], 'idx': f['idx']} for f in functions_orig]

    candidate_pool_intermediary = []
    
    # Add top min(N, K) by A descending (then B descending for ties)
    functions_temp.sort(key=lambda f: (-f['A'], -f['B']))
    for i in range(min(N, K)): # N is len(functions_temp)
        candidate_pool_intermediary.append(functions_temp[i])
        
    # Add top min(N, K) by B descending (then A descending for ties)
    functions_temp.sort(key=lambda f: (-f['B'], -f['A']))
    for i in range(min(N, K)):
        candidate_pool_intermediary.append(functions_temp[i])

    # Create unique candidate pool using original indices
    unique_candidates_dict = {}
    for f_dict in candidate_pool_intermediary:
        unique_candidates_dict[f_dict['idx']] = f_dict
    candidate_pool_unique = list(unique_candidates_dict.values())
    
    # Initialize max_val. Since all A_i, B_i >= 1, and K >= 1,
    # the smallest possible result is f(1)=A*1+B >= 1*1+1=2.
    # So -float('inf') or even 0 would be fine.
    max_overall_value = -float('inf') 

    if K == 0: # Though K >= 1 as per constraints usually.
        print(1) # Applying zero functions to 1 results in 1.
        return
    
    # If K=1 and N=0, candidate_pool_unique can be empty.
    # Constraints: 1 <= N, 1 <= K <= min(N,10). So N >= K >= 1.
    # Thus, candidate_pool_unique will contain at least min(N,K) = K elements.
    # Unless N is very small, e.g. N=1, K=1. Pool size is 1.
    # The number of functions to choose is K.
    # The list to pick from is candidate_pool_unique. Its length is at least K if N >= K.
    
    # If candidate_pool_unique is empty, but K > 0. This shouldn't happen with N>=1, K>=1.
    if not candidate_pool_unique and K > 0:
         # This case should not be reached given problem constraints.
         # If it were, the answer would be based on initial max_overall_value.
         # Smallest possible result is 2 (e.g. f(x)=x+1, K=1, f(1)=2).
         # If K > 0 and no functions, it's ill-defined, but constraints prevent this.
         pass


    for current_K_indices_in_pool in combinations(range(len(candidate_pool_unique)), K):
        current_K_functions_selection = [candidate_pool_unique[i] for i in current_K_indices_in_pool]
        
        # Define comparison for sorting the selected K functions
        def compare_selected_functions(f1_dict, f2_dict):
            A1, B1 = f1_dict['A'], f1_dict['B']
            A2, B2 = f2_dict['A'], f2_dict['B']

            # Case 1: One has A=1, the other A>1
            if A1 > 1 and A2 == 1: return -1  # f1 (A>1) is outer to f2 (A=1)
            if A1 == 1 and A2 > 1: return 1   # f2 (A>1) is outer to f1 (A=1)
            
            # Case 2: Both A=1
            if A1 == 1 and A2 == 1: return 0  # Order equivalent
            
            # Case 3: Both A>1. Compare B/(A-1) ratios.
            # f1 is outer to f2 if B1/(A1-1) < B2/(A2-1)
            # This is B1*(A2-1) < B2*(A1-1) to avoid float arithmetic and division by zero.
            val_f1_cmp = B1 * (A2 - 1)
            val_f2_cmp = B2 * (A1 - 1)
            
            if val_f1_cmp < val_f2_cmp: return -1
            if val_f1_cmp > val_f2_cmp: return 1
            
            # Tie-breaking (optional, mainly for stable sort, does not affect max value if truly equivalent)
            # Could use original indices for a fully deterministic sort:
            # if f1_dict['idx'] < f2_dict['idx']: return -1
            # if f1_dict['idx'] > f2_dict['idx']: return 1
            return 0

        # sorted_list is (p_1, p_2, ..., p_K) after sorting.
        # p_1 is "most outer", p_K is "most inner".
        sorted_list_for_application = sorted(current_K_functions_selection, key=cmp_to_key(compare_selected_functions))
        
        current_composed_value = 1
        # Apply functions: f_{p_1}(f_{p_2}(...f_{p_K}(1)...))
        # Innermost function is p_K (last in sorted_list_for_application).
        # Outermost is p_1 (first in sorted_list_for_application).
        # Iterate application from innermost to outermost: apply p_K, then p_{K-1}, ..., then p_1.
        # This means iterating sorted_list_for_application from K-1 down to 0.
        for i in range(K - 1, -1, -1):
            f_params = sorted_list_for_application[i]
            current_composed_value = f_params['A'] * current_composed_value + f_params['B']
        
        if current_composed_value > max_overall_value:
            max_overall_value = current_composed_value
            
    print(max_overall_value)

solve()