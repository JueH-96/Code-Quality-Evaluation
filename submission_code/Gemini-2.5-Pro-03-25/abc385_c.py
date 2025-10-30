# YOUR CODE HERE
import sys

def solve():
    # Read the number of buildings N from standard input.
    N = int(sys.stdin.readline())
    
    # According to constraints, N is at least 1.
    # Handle the edge case N=0, although it shouldn't occur based on constraints.
    if N == 0:
        print(0)
        return

    # Read the heights of the N buildings from standard input.
    # The heights are space-separated integers on a single line.
    # map(int, ...) converts each string part into an integer.
    # list(...) converts the map object into a list.
    H = list(map(int, sys.stdin.readline().split()))

    # Initialize the maximum length found so far.
    # Choosing a single building is always possible, so the minimum answer is 1.
    # This initialization also correctly handles the N=1 case, as the loops below won't execute.
    max_k = 1
    
    # If N=1, the maximum length is 1. We can return immediately.
    if N == 1:
       print(1)
       return

    # Iterate through all possible differences 'd' between the indices of chosen buildings.
    # The difference 'd' must be at least 1 (since buildings are distinct).
    # The maximum possible difference is N-1 (e.g., between buildings at index 0 and N-1).
    # The loop range is [1, N-1).
    for d in range(1, N): 
        # For each difference 'd', we use a dictionary to keep track of the lengths of 
        # arithmetic progressions (APs) found so far that have this common difference 'd'.
        # The dictionary maps the ending index 'j' of an AP to its length.
        # This dictionary is specific to the current difference 'd' and effectively implements
        # dynamic programming with O(N) space per difference 'd'. The total space complexity is O(N).
        current_d_lengths = {} 
        
        # Iterate through all possible ending indices 'j' for an AP with difference 'd'.
        # An AP must have at least two terms to define a difference d > 0.
        # The pair of indices (i, j) defines the difference d = j - i.
        # The minimum value for j is d (corresponding to i=0).
        # The maximum value for j is N-1 (the last building).
        # The loop range is [d, N).
        for j in range(d, N): 
            # Calculate the index 'i' of the potential previous building in the AP.
            i = j - d 
            
            # Check if the buildings at indices i and j have the same height.
            # This is a necessary condition for them to be part of the same AP satisfying the problem conditions.
            if H[i] == H[j]:
                # If the heights match, we can potentially extend an existing AP ending at 'i'
                # or start a new AP of length 2 consisting of buildings 'i' and 'j'.
                
                # Retrieve the length of the AP ending at index 'i' with the same difference 'd'.
                # We use the dictionary's `.get(key, default)` method.
                # `current_d_lengths.get(i, 1)` returns `current_d_lengths[i]` if 'i' is a key in the dictionary.
                # If 'i' is not a key, it means no AP ending at 'i' with difference 'd' and length > 1 was found previously.
                # In this case, `get` returns the default value 1. This represents the base case:
                # building 'i' itself forms an AP of length 1.
                prev_len = current_d_lengths.get(i, 1)
                
                # The length of the AP ending at index 'j' is one greater than the length of the AP ending at 'i'.
                current_len = prev_len + 1
                
                # Store this calculated length in the dictionary, associating it with the ending index 'j'.
                # This updates the DP state for index 'j' and difference 'd'.
                current_d_lengths[j] = current_len
                
                # Compare the newly found AP length `current_len` with the maximum length `max_k` found so far across all differences and indices.
                # Update `max_k` if `current_len` is greater.
                max_k = max(max_k, current_len)
            
            # If H[i] != H[j], the buildings at indices i and j cannot be consecutive elements
            # in a constant-height AP with difference 'd'. The sequence breaks.
            # Any AP ending at 'j' with difference 'd' would have to start at 'j' itself, having length 1.
            # We don't need to explicitly store length 1 in the dictionary `current_d_lengths`.
            # The logic using `.get(key, 1)` correctly handles this: if `j` is later used as an index `i'`
            # for calculating the length for `j+d`, `current_d_lengths.get(j, 1)` will return 1, effectively
            # starting a new sequence of length 2 from `(j, j+d)` if `H[j] == H[j+d]`.

    # After iterating through all possible differences 'd' and all possible ending indices 'j',
    # the variable `max_k` holds the maximum length found among all valid arithmetic progressions
    # satisfying the problem conditions. Print the final result.
    print(max_k)

# Execute the solve function to run the program.
solve()