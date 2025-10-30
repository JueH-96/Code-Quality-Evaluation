import sys
import bisect

def solve():
    # Read N
    # Use strip() to handle potential trailing newline
    N = int(sys.stdin.readline().strip())

    # Read A as a list of integers
    A = list(map(int, sys.stdin.readline().strip().split()))

    # Calculate total count of valid kagamimochi
    total_count = 0
    
    # Iterate through each mochi, considering it as the bottom one (index j)
    # For each bottom mochi A[j], we count how many mochi A[i] can be placed on top.
    for j in range(N):
        # A[j] is the size of the bottom mochi
        # We need to find the number of possible top mochi A[i] (where i != j)
        # such that A[i] <= A[j] / 2
        
        # Calculate the maximum allowed size for the top mochi using integer division
        target_value = A[j] // 2
        
        # Since the list A is sorted, all elements A[i] satisfying A[i] <= target_value
        # will appear at the beginning of the list.
        # bisect.bisect_right(A, value) returns the index `p` such that
        # all elements A[0]...A[p-1] are <= value, and all elements A[p]...A[N-1] are > value.
        # Thus, `p` is the count of elements in A that are less than or equal to target_value.
        count_valid_tops = bisect.bisect_right(A, target_value)
        
        # The count_valid_tops represents the number of indices `i` such that A[i] <= target_value.
        # We are counting pairs of distinct mochi instances (indices i, j) where i != j
        # and A[i] <= A[j] / 2.
        # Since A[k] >= 1 for all k, A[j] is positive. A[j] // 2 is strictly less than A[j]
        # for A[j] >= 1.
        # This means A[j] itself can never be less than or equal to A[j] // 2.
        # Consequently, the index `j` is never included in the set of indices `i`
        # counted by `bisect.bisect_right(A, target_value)`.
        # The count_valid_tops therefore already represents the number of *other* mochi A[i]
        # (with i != j) that can be placed on top of A[j].
        
        total_count += count_valid_tops

    # Print the total count of different kinds of kagamimochi
    print(total_count)

# Execute the solve function
solve()