import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Base cases for N=1 or N=2
    # If N=1, there's 1 AP: (A_0)
    # If N=2, there are 3 APs: (A_0), (A_1), (A_0, A_1)
    # These are N * (N + 1) // 2
    if N <= 2:
        print(N * (N + 1) // 2)
        return

    # Initialize total_ans with count of all single-element APs.
    # Each (A_i) is an AP of length 1. There are N such APs.
    total_ans = N 
    
    # current_len tracks the length of the arithmetic progression ending at current index i
    # that maintains a consistent common difference from its start.
    # It will be at least 2 when considering elements from index 1 onwards.
    current_len = 0 
    
    # prev_diff stores the common difference calculated from the previous pair (A[i-1] - A[i-2])
    # or the first pair (A[1] - A[0]) when i starts from 1.
    prev_diff = None 

    # Iterate from the second element (index 1) to compare with previous elements
    for i in range(1, N):
        # Calculate the difference between the current element and the one before it
        diff = A[i] - A[i-1]
        
        # Check if this difference continues the previous arithmetic progression
        if diff == prev_diff:
            # The current element extends the existing AP segment.
            # E.g., if (A_k, A_{k+1}, ..., A_{i-1}) was an AP of length `current_len`,
            # then (A_k, A_{k+1}, ..., A_{i-1}, A_i) is now an AP of length `current_len + 1`.
            current_len += 1
        else:
            # The common difference has changed, or this is the very first pair (i=1).
            # A new arithmetic progression starts with (A[i-1], A[i]).
            # This is an AP of length 2.
            current_len = 2 
            # Update prev_diff to the new common difference for future comparisons.
            prev_diff = diff
        
        # Add the number of new APs ending at index i.
        # If current_len is k, it means (A[i-k+1], ..., A[i]) is an AP.
        # This contributes (k-1) new valid (l,r) pairs ending at 'i'.
        # For example, if current_len=2, it means (A[i-1],A[i]) is an AP. Add 1.
        # If current_len=3, it means (A[i-2],A[i-1],A[i]) is an AP. Add 2.
        # The (A[i]) AP (length 1) is already counted in the initial `total_ans = N`.
        total_ans += (current_len - 1)
        
    print(total_ans)

solve()