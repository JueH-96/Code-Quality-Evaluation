import sys
import itertools
from functools import cmp_to_key

# It's good practice to wrap the main logic in a function.
def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use a faster method for reading input, as N can be large.
    def get_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    try:
        # Read problem parameters and function coefficients
        N, K = get_ints()
        funcs = []
        for i in range(N):
            A, B = get_ints()
            # Store functions as dictionaries for clarity, including their original index
            funcs.append({'A': A, 'B': B, 'id': i})
    except (IOError, ValueError):
        # Handle potential empty input during local testing
        return

    # The problem statement guarantees 1 <= K
    
    # Define the comparison function for sorting by C-value.
    # To sort in descending order of C_i = B_i / (A_i - 1), we compare B_i(A_j-1) and B_j(A_i-1).
    # A positive return value from cmp(i, j) means i is "greater" than j.
    def compare_C(item1, item2):
        A1, B1 = item1['A'], item1['B']
        A2, B2 = item2['A'], item2['B']
        val = B1 * (A2 - 1) - B2 * (A1 - 1)
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0

    # This set will store the indices of candidate functions
    candidate_ids = set()
    
    # The number of candidates to pick from each category
    limit = min(K, N)

    # Sort by C-value descending to find candidates for the start of the composition chain
    c_sorted_funcs = sorted(funcs, key=cmp_to_key(compare_C), reverse=True)
    for i in range(limit):
        candidate_ids.add(c_sorted_funcs[i]['id'])
    
    # Add candidates from the other end of the C-value spectrum (for the end of the chain)
    for i in range(limit):
        candidate_ids.add(c_sorted_funcs[N - 1 - i]['id'])
        
    # Sort by A+B to find candidates that give a large initial value
    v_sorted_funcs = sorted(funcs, key=lambda f: f['A'] + f['B'], reverse=True)
    for i in range(limit):
        candidate_ids.add(v_sorted_funcs[i]['id'])

    # Create the final pool of function objects from the collected indices
    candidate_funcs_map = {f['id']: f for f in funcs}
    pool = [candidate_funcs_map[cid] for cid in candidate_ids]

    # Sort the pool by C-value. This is a key optimization.
    sorted_pool = sorted(pool, key=cmp_to_key(compare_C), reverse=True)
    
    max_val = -float('inf')
    
    # Iterate through all K-sized combinations from the candidate pool.
    # The pool size is at most 3K, which is small enough for this to be efficient.
    for combination in itertools.combinations(sorted_pool, K):
        # itertools.combinations preserves the order of the input iterable.
        # Since sorted_pool is sorted by C-value, each combination is also sorted.
        # The optimal permutation applies functions in increasing order of C-value,
        # which is the reverse of our sorted combination.
        current_val = 1
        for f in reversed(combination):
            current_val = f['A'] * current_val + f['B']
        
        if current_val > max_val:
            max_val = current_val
                
    print(max_val)

solve()