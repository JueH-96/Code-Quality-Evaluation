import sys

def solve():
    # Read N and X from the first line of input.
    # map(int, ...) converts string tokens to integers.
    N, X = map(int, sys.stdin.readline().split())
    
    # Read the array A from the second line of input.
    # Convert string tokens to integers and store them in a list.
    # A will be 0-indexed in Python (A[0] to A[N-1]).
    # The problem asks for 1-based indices in the output (1 to N).
    A = list(map(int, sys.stdin.readline().split()))

    # Iterate through each element A[i] to serve as the first number in the triplet.
    # `i` is the 0-based index.
    # The loop runs up to N-2 because we need at least two more elements (A[p] and A[j])
    # after A[i] to form a triplet. So, `i` can be at most N-3.
    for i in range(N - 2):
        
        # Calculate the sum that the remaining two elements (A[p] and A[j]) must achieve.
        target_two_sum = X - A[i]
        
        # Initialize a hash map (Python dictionary) for the "two-sum" subproblem.
        # This map will store {value: index} for elements encountered in the inner loop.
        # It's reset for each new `i` to ensure `p > i`.
        seen_elements = {} 
        
        # Iterate through elements A[j] to serve as the third number in the triplet.
        # `j` is the 0-based index.
        # We start `j` from `i + 1` to ensure `j > i`.
        # The loop runs up to N-1 (inclusive) to cover all remaining elements.
        for j in range(i + 1, N):
            
            # Calculate the value required for the second number (A[p]).
            required_p_val = target_two_sum - A[j]
            
            # Check if `required_p_val` has been seen before in the current window of elements
            # (i.e., among A[i+1], ..., A[j-1]).
            if required_p_val in seen_elements:
                # If found, `p` is the index of the element A[p] that makes the sum.
                p = seen_elements[required_p_val]
                
                # We have found a triplet (A[i], A[p], A[j]) that sums to X.
                # The 0-based indices are i, p, j.
                # The conditions `1 <= i < j < k <= N` must be met for 1-based indices.
                # Our current 0-based indices satisfy `i < p < j` because:
                # 1. `i < p`: `p` is an index from `seen_elements`, populated from `i+1` onwards.
                # 2. `p < j`: `p` was added to `seen_elements` at an index `p` strictly less than the current `j`.
                # So, `(i+1)`, `(p+1)`, `(j+1)` forms a valid 1-based triplet.
                
                # Print the 1-based indices and terminate the program.
                print(i + 1, p + 1, j + 1)
                return
            
            # If `required_p_val` was not found, add the current element A[j] to `seen_elements`.
            # This makes A[j] available for future lookups as a potential `A[p]` for later `j'` values.
            # We add A[j] *after* checking for `required_p_val` to ensure that A[j] is not
            # used with itself (i.e., `p` cannot be equal to `j`).
            seen_elements[A[j]] = j
            
    # If the loops complete without finding any valid triplet, print -1.
    print("-1")

# Call the solve function to execute the program logic.
solve()