# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    # Constraints state N >= 1. Handle N=0 case just in case, although unlikely.
    if N == 0: 
        print() 
        return
    
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Define the modulus
    MOD = 998244353

    # Initialize the DP structure.
    # dp[j] will store DP state information for arithmetic subsequences (APs) ending at index j.
    # It is implemented as a list of dictionaries. Each dictionary `dp[j]` maps a common difference `d` 
    # to another dictionary.
    # The inner dictionary `dp[j][d]` maps a length `k` to the count of APs of length k 
    # with common difference `d` ending exactly at index j.
    dp = [{} for _ in range(N)] 

    # Initialize the result array C.
    # C[k] will store the total count of arithmetic subsequences of length k found in A.
    # The array size is N+1 to store counts for lengths 1 through N. Index 0 is unused.
    C = [0] * (N + 1)
    
    # Base case: Any single element A[i] forms an arithmetic subsequence of length 1.
    # There are N such subsequences.
    # This check is technically redundant given N >= 1 constraint, but included for clarity.
    if N >= 1:
      C[1] = N

    # Dynamic Programming calculation
    # Iterate through each index j from 0 to N-1. A[j] serves as the potential last element of an AP.
    for j in range(N):
        # For each j, iterate through all previous indices i (0 <= i < j). 
        # A[i] serves as the potential second-to-last element of an AP ending at A[j].
        for i in range(j):
            # Calculate the common difference `d` required if A[i] and A[j] are consecutive terms in an AP.
            d = A[j] - A[i]

            # The pair (A[i], A[j]) itself forms an arithmetic subsequence of length 2.
            # Every pair (i, j) with i < j contributes one AP of length 2.
            # Increment the total count for length 2 APs.
            # This check `if N >= 2:` is implicitly true if this inner loop runs (since j >= 1 implies N >= 2).
            if N >= 2: 
                 C[2] = (C[2] + 1) % MOD
            
            # Initialize the count for the base case AP (A[i], A[j]) of length 2.
            # This specific pair contributes 1 to the count of length 2 APs ending at j with difference d.
            count_for_len2 = 1 
            
            # Check if there are any APs ending at index i with the same common difference d.
            # These APs can potentially be extended by A[j].
            # `d in dp[i]` checks if difference `d` was encountered ending at index `i`.
            if d in dp[i]:
                 # If yes, iterate through items (length `k`, count `c`) stored in dp[i][d].
                 # `dp[i][d].items()` provides pairs of (length, count) for APs ending at `i` with difference `d`.
                 for length_k, count_c in dp[i][d].items():
                     # An AP of length `length_k` ending at `i` can be extended by `A[j]`
                     # to form a new AP of length `new_length = length_k + 1`.
                     new_length = length_k + 1
                     
                     # Ensure the length of the extended AP does not exceed N.
                     if new_length <= N:
                         # Update the DP state for index `j`: dp[j][d][new_length].
                         # The number of such new APs is `count_c`.
                         # We need to add `count_c` to any existing count for this state.
                         # `setdefault(d, {})` ensures `dp[j][d]` dictionary exists before access.
                         # `get(new_length, 0)` safely retrieves the current count for `new_length` or 0 if it doesn't exist yet.
                         current_new_len_count = dp[j].setdefault(d, {}).get(new_length, 0)
                         dp[j][d][new_length] = (current_new_len_count + count_c) % MOD
                         
                         # Also, update the total count `C[new_length]` by adding `count_c`.
                         # Each extended AP contributes to the total count for its length.
                         C[new_length] = (C[new_length] + count_c) % MOD
            
            # After considering extensions, update the count for length 2 APs ending at j with difference d.
            # We add the `count_for_len2` (which is 1) representing the base pair (A[i], A[j]).
            # Use `setdefault` and `get` for safe dictionary access and updates.
            current_len2_count = dp[j].setdefault(d, {}).get(2, 0)
            dp[j][d][2] = (current_len2_count + count_for_len2) % MOD

    # After iterating through all pairs (i, j), the array C contains the total counts.
    # Print the final counts C[1] through C[N], separated by spaces.
    # `C[1:]` slices the list to get elements from index 1 to N.
    # `*` unpacks the list elements as arguments to print.
    print(*(C[1:]))

# Execute the solve function
solve()