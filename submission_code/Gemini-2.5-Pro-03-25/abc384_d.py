# YOUR CODE HERE
import sys

# Attempt to import SortedList from sortedcontainers library.
# This library provides an efficient implementation of a sorted list,
# which behaves like a list but maintains sorted order and allows
# O(log N) operations for additions, deletions, and searches.
try:
    from sortedcontainers import SortedList
except ImportError:
    # If the library is not available, print an error message to stderr and exit.
    # In competitive programming environments like Codeforces or AtCoder, external 
    # libraries are typically not allowed unless they are standard. If this code
    # needs to run in such an environment, one would need to either bundle the
    # library source code (if allowed) or implement the required data structure 
    # (like a balanced binary search tree) using standard library components.
    sys.stderr.write("Error: sortedcontainers library not found. This solution requires it.
")
    sys.exit(1) # Exit with a non-zero status to indicate an error

def main():
    # Read N and S from standard input.
    # N: the period of the infinite sequence A.
    # S: the target sum.
    N, S = map(int, sys.stdin.readline().split())
    
    # Read the first N terms of the sequence A.
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the sum of one period (A_1 + ... + A_N).
    T = sum(A)

    # Constraints state A_i >= 1 and N >= 1. Therefore, T must be >= N >= 1.
    # T cannot be 0 under the problem constraints.

    # Construct sequence B by concatenating A with itself (A_1..A_N, A_1..A_N).
    # B has length 2N. Indices are 0 to 2N-1.
    # Considering subsequences within B allows us to handle subsequences of A
    # that span across the period boundary N. Any contiguous subsequence of A
    # of length up to N is equivalent to some contiguous subsequence within B.
    B = A * 2 
    
    # Compute prefix sums P' for B.
    # P'[k] = sum(B[0...k-1]). P' array has length 2N+1. P'[0] = 0.
    # P'[k] stores the sum of the first k elements of B.
    prefix_sum_B = [0] * (2 * N + 1)
    for i in range(2 * N):
        prefix_sum_B[i+1] = prefix_sum_B[i] + B[i]

    # Data Structure (DS): A dictionary mapping (remainder modulo T) to 
    # (SortedList containing prefix sum values P'[k]).
    # This structure efficiently stores and queries prefix sums P'[k] 
    # corresponding to indices k within the current sliding window.
    DS = {} 

    # Initialize DS with P'[0] = 0.
    P_k_val = prefix_sum_B[0] # This is P'[0]
    
    # Calculate remainder modulo T. Since T >= 1, this is well-defined.
    # Python's % operator handles negative numbers correctly for modulo arithmetic.
    rem = P_k_val % T
    
    # Initialize the SortedList for this remainder with P'[0].
    DS[rem] = SortedList([P_k_val]) 

    # Iterate through possible end points of subsequences within B.
    # 'r' represents the index for the prefix sum P'[r].
    # P'[r] is the sum of elements B[0...r-1].
    # The subsequence ends at index r-1 in B.
    for r in range(1, 2 * N + 1):
        
        P_r_val = prefix_sum_B[r] # Current prefix sum P'[r]
        
        # Query Step:
        # We are looking for a contiguous subsequence sum S.
        # Any such sum can be represented as S = q*T + Sigma, where q >= 0 is an integer,
        # and Sigma is the sum of a contiguous subsequence of A with length at most N.
        # Sigma can be written as P'[r] - P'[k] for some k.
        # The condition S = q*T + Sigma implies S >= Sigma and S % T == Sigma % T.
        # Substituting Sigma = P'[r] - P'[k], we get:
        # S >= P'[r] - P'[k]  =>  P'[k] >= P'[r] - S
        # S % T == (P'[r] - P'[k]) % T  =>  P'[k] % T == (P'[r] - S) % T
        # We need to find if there exists a P'[k] in the current window satisfying these two conditions.
        
        # Calculate the target remainder P'[k] should have.
        target_rem = (P_r_val - S) % T
        # Calculate the minimum value P'[k] must have.
        threshold = P_r_val - S

        # Check if there are any prefix sums stored in DS with the target remainder.
        if target_rem in DS:
            sl = DS[target_rem] # Get the SortedList associated with target_rem.
            
            # Find the smallest value P_val in sl such that P_val >= threshold.
            # `bisect_left(threshold)` returns the index where `threshold` would be inserted
            # while maintaining sorted order. The element at this index, if it exists,
            # is the smallest element in `sl` that is >= `threshold`.
            idx = sl.bisect_left(threshold)
            
            # If `idx` is less than the length of `sl`, it means we found such an element.
            if idx < len(sl):
                # A P'[k] value satisfying both conditions has been found.
                # The sliding window logic ensures this P'[k] corresponds to an index k
                # that is within the valid window range [max(0, r-N), r-1].
                # Thus, a contiguous subsequence with sum S exists.
                print("Yes")
                return # Exit the function successfully.

        # Update window state for the next iteration (r+1):
        # Add the current prefix sum P'[r] to the data structure DS.
        # This value P'[r] will become available for queries in future iterations
        # when it falls within the sliding window.
        current_rem = P_r_val % T
        if current_rem not in DS:
            # If this remainder is not yet a key in DS, initialize a new SortedList.
            DS[current_rem] = SortedList()
        # Add P'[r] to the SortedList for its remainder.
        DS[current_rem].add(P_r_val)

        # Remove the prefix sum P'[k_out] that slides out of the window.
        # The window contains prefix sums P'[k] for k in [max(0, r-N), r-1].
        # When moving from r to r+1, the index k = r - N becomes too old and must be removed.
        if r >= N:
             k_out = r - N # Index of the prefix sum to remove.
             P_k_out_val = prefix_sum_B[k_out]
             rem_out = P_k_out_val % T
             
             # Check if the remainder exists as a key in DS before attempting removal.
             if rem_out in DS:
                 # Remove one instance of the value P_k_out_val from the SortedList.
                 DS[rem_out].remove(P_k_out_val) 
                 # If the SortedList becomes empty after removal, delete the key from DS.
                 if not DS[rem_out]: 
                      del DS[rem_out]

    # If the loop completes without finding any P'[k] satisfying the conditions for any r,
    # it means no contiguous subsequence sum equals S.
    print("No")


# Standard boilerplate for calling the main function when the script is executed.
if __name__ == '__main__':
    main()