# YOUR CODE HERE
import sys
import bisect

# Function to perform binary search to find the first index k in [low_idx, high_idx]
# such that check_func(k) is True. Returns -1 if no such k exists.
# Uses iterative binary search.
def binary_search_first(low_idx, high_idx, check_func):
    """Finds the smallest index k in [low_idx, high_idx] such that check_func(k) is True."""
    k_res = -1 # Initialize result to -1 (not found)
    current_low = low_idx
    current_high = high_idx
    while current_low <= current_high:
        k_mid = (current_low + current_high) // 2 # Calculate midpoint index
        if check_func(k_mid):
            # If check_func is true for k_mid, it might be the first such index.
            # Store it as a potential result and try searching in the left half (including k_mid)
            # to find an even smaller index.
            k_res = k_mid
            current_high = k_mid - 1 # Try smaller index
        else:
            # If check_func is false for k_mid, the first valid index must be larger.
            # Search in the right half.
            current_low = k_mid + 1 # Need larger index
    return k_res

# Function to perform binary search to find the last index k in [low_idx, high_idx]
# such that check_func(k) is True. Returns -1 if no such k exists.
# Uses iterative binary search.
def binary_search_last(low_idx, high_idx, check_func):
    """Finds the largest index k in [low_idx, high_idx] such that check_func(k) is True."""
    k_res = -1 # Initialize result to -1 (not found)
    current_low = low_idx
    current_high = high_idx
    while current_low <= current_high:
        k_mid = (current_low + current_high) // 2 # Calculate midpoint index
        if check_func(k_mid):
            # If check_func is true for k_mid, it might be the last such index.
            # Store it as a potential result and try searching in the right half (including k_mid)
            # to find an even larger index.
            k_res = k_mid
            current_low = k_mid + 1 # Try larger index
        else:
            # If check_func is false for k_mid, the last valid index must be smaller.
            # Search in the left half.
            current_high = k_mid - 1 # Need smaller index
    return k_res

def solve():
    # Optimize I/O reading speed
    read_ints = lambda: map(int, sys.stdin.readline().split())
    read_str = lambda: sys.stdin.readline().strip()

    N, Q = read_ints() # Read problem size N and number of queries Q
    S = read_str() # Read the string S

    # Precompute prefix sums for counts of '1's and '2's
    # P1[i] stores the count of '1's in the prefix S[0...i-1]
    # P2[i] stores the count of '2's in the prefix S[0...i-1]
    P1 = [0] * (N + 1)
    P2 = [0] * (N + 1)
    
    for i in range(N):
        # Copy previous prefix sum value
        P1[i+1] = P1[i]
        P2[i+1] = P2[i]
        # Increment count if the current character matches
        if S[i] == '1':
            P1[i+1] += 1
        elif S[i] == '2':
            P2[i+1] += 1

    # Store all 0-based indices where '/' character appears in S
    slash_indices = [i for i, char in enumerate(S) if char == '/']

    # Process Q queries
    results_buffer = [] # Use a buffer to store results for potentially faster final output
    for _ in range(Q):
        L, R = read_ints() # Read query range L, R (1-based)
        # Convert query L, R to 0-based indices [L_idx, R_idx] for the substring S[L_idx...R_idx]
        L_idx = L - 1
        R_idx = R - 1

        # Find the range of indices within the `slash_indices` list that correspond to '/' characters
        # located within the substring range [L_idx, R_idx].
        # `bisect_left` finds the first index >= L_idx.
        start_slash_k = bisect.bisect_left(slash_indices, L_idx)
        # `bisect_right` finds the first index > R_idx.
        end_slash_k = bisect.bisect_right(slash_indices, R_idx)
        
        # The relevant indices in `slash_indices` are `start_slash_k` through `end_slash_k - 1`.
        
        # Check if there are any '/' characters within the given substring range [L, R].
        if start_slash_k >= end_slash_k:
            # If no slashes are found, an 11/22 string subsequence cannot be formed (minimum length is 1 for '/').
            results_buffer.append("0")
            continue

        # An 11/22 string has the form 1...1 / 2...2 with m ones and m twos. Its length is 2m+1.
        # We want to find the maximum possible value of m.
        
        # Initialize the best m found so far. If slashes exist, m=0 is always possible (resulting in '/'), length 1.
        current_best_m = 0 
        
        # Perform binary search on the possible values of m.
        # The minimum possible m is 0. Maximum possible m is roughly N/2.
        low_m = 0
        high_m = N // 2 
        
        # `current_best_m` stores the maximum `mid_m` for which the check passed.
        
        while low_m <= high_m:
            mid_m = (low_m + high_m) // 2 # Calculate midpoint m value to check
            
            # The base case m=0 is correctly handled by initializing current_best_m = 0
            # and the logic below which will find m=0 is possible if any slash exists.

            # Define check functions passed to binary search helpers. These functions capture
            # the conditions required for a given m (`mid_m`).
            
            # `check_k_first(k_idx)` returns True if the count of '1's before the slash at `slash_indices[k_idx]`
            # is sufficient (>= mid_m). The count is for the range S[L_idx ... j-1] where j = slash_indices[k_idx].
            # This count is P1[j] - P1[L_idx].
            check_k_first = lambda k_idx: P1[slash_indices[k_idx]] - P1[L_idx] >= mid_m
            
            # `check_k_last(k_idx)` returns True if the count of '2's after the slash at `slash_indices[k_idx]`
            # is sufficient (>= mid_m). The count is for the range S[j+1 ... R_idx] where j = slash_indices[k_idx].
            # This count is P2[R_idx + 1] - P2[j + 1].
            check_k_last = lambda k_idx: P2[R_idx + 1] - P2[slash_indices[k_idx] + 1] >= mid_m

            # Find the smallest index `k_first` in the relevant range `[start_slash_k, end_slash_k - 1]`
            # such that `check_k_first` condition holds.
            k_first = binary_search_first(start_slash_k, end_slash_k - 1, check_k_first)

            # Find the largest index `k_last` in the relevant range `[start_slash_k, end_slash_k - 1]`
            # such that `check_k_last` condition holds.
            k_last = binary_search_last(start_slash_k, end_slash_k - 1, check_k_last)
            
            # A valid 11/22 string with parameter `mid_m` can be formed if we can find *some* slash index `j`
            # that satisfies both conditions. This is possible if and only if there exists a `k` such that
            # `k_first <= k <= k_last`. This condition simplifies to `k_first <= k_last`, assuming both `k_first`
            # and `k_last` were found (i.e., are not -1).
            if k_first != -1 and k_last != -1 and k_first <= k_last:
                 # If `mid_m` is achievable, we potentially found a new maximum m. Store it.
                 # Then, try searching for an even larger m in the upper half of the search space.
                 current_best_m = mid_m 
                 low_m = mid_m + 1 
            else:
                 # If `mid_m` is not achievable, the maximum possible m must be smaller.
                 # Search in the lower half of the search space.
                 high_m = mid_m - 1 

        # After the binary search completes, `current_best_m` holds the maximum value of m achievable.
        # The maximum length of the 11/22 subsequence is 2*m + 1.
        results_buffer.append(str(2 * current_best_m + 1))

    # Print all collected results, separated by newlines.
    print("
".join(results_buffer))

# Call the main function to run the solution logic.
solve()