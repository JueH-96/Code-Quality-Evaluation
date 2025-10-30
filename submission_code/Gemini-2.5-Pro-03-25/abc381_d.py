import sys

def solve():
    """
    Solves the problem of finding the maximum length of a contiguous subarray
    that is a 1122 sequence. Reads input from stdin and prints output to stdout.
    """
    N = int(sys.stdin.readline())
    
    # Handle edge cases for N=0 or N=1. Constraints state N >= 1, but it's safe practice.
    # The empty sequence has length 0 and is a 1122 sequence.
    # If N=0, the array is empty, max length is 0.
    # If N=1, the array has one element. No subarray of even length > 0 is possible. Max length is 0.
    if N <= 1:
        # Read the potentially empty list of numbers even if N=0 or N=1 to consume input
        if N == 1:
             _ = sys.stdin.readline() # Read the single element
        print(0)
        return
        
    A = list(map(int, sys.stdin.readline().split()))

    def solve_sliding_window(C):
        """
        Helper function using a sliding window approach to find the maximum length 
        of a contiguous subarray of C containing only distinct positive integers.
        
        Args:
            C: A list of integers, where positive integers represent values from pairs
               and 0 represents invalid pairs (non-equal elements).
        
        Returns:
            The maximum length found.
        """
        max_len = 0
        p = 0  # Start index of the current valid window (0-based)
        # Dictionary to store the last seen index of each positive value encountered.
        # Format: {value: last_index_q}
        last_pos = {} 
        
        for q in range(len(C)):
            current_val = C[q]
            
            if current_val == 0:
                # A zero indicates an invalid pair (A[idx] != A[idx+1]).
                # This breaks the current potential 1122 sequence structure.
                # Any new valid window must start after this index q.
                p = q + 1
                # We don't need to clear last_pos. The check `prev_idx >= p` later
                # ensures that we only consider previous occurrences within the
                # current window's valid range defined by p.
            else: # current_val > 0, indicating a valid pair (A[idx] == A[idx+1])
                if current_val in last_pos:
                    prev_idx = last_pos[current_val]
                    # Check if the previous occurrence of current_val is within the current window scope [p, q).
                    # An index `prev_idx` is within the scope if `prev_idx >= p`.
                    if prev_idx >= p:
                        # Duplicate value found within the current window range [p, q].
                        # To maintain the distinctness property required by condition 3,
                        # we must slide the window start `p` to the position immediately 
                        # after the previous occurrence (`prev_idx + 1`).
                        p = prev_idx + 1
                
                # Update the last seen position of current_val to the current index q
                last_pos[current_val] = q
                
                # The current valid window is C[p...q]. Its length is q - p + 1.
                # By construction, this window contains distinct positive integers.
                current_window_len = q - p + 1
                # Update the maximum length found so far.
                max_len = max(max_len, current_window_len)
                
        return max_len

    # Construct sequence C_odd.
    # Each element C_odd[k] corresponds to the pair (A[2k], A[2k+1]) starting at an even index 2k.
    # If A[2k] == A[2k+1], C_odd[k] stores the value A[2k]. Otherwise, it stores 0.
    C_odd = []
    M1 = N // 2 # Number of pairs starting at even indices 0, 2, ..., 2*(M1-1)
    for k in range(M1):
        idx1 = 2*k
        idx2 = 2*k + 1
        # Check boundary conditions implicitly handled by range(M1) which ensures idx2 < N
        if A[idx1] == A[idx2]:
            C_odd.append(A[idx1]) 
        else:
            C_odd.append(0)

    # Construct sequence C_even.
    # Each element C_even[k] corresponds to the pair (A[2k+1], A[2k+2]) starting at an odd index 2k+1.
    # If A[2k+1] == A[2k+2], C_even[k] stores the value A[2k+1]. Otherwise, it stores 0.
    C_even = []
    # Number of pairs starting at odd indices 1, 3, ...
    # The last possible pair starts at index N-2 (if N even) or N-3 (if N odd). Its indices are (N-2, N-1) or (N-3, N-2).
    # The loop range ensures 2k+2 < N.
    M2 = (N - 1) // 2 
    for k in range(M2):
        idx1 = 2*k + 1
        idx2 = 2*k + 2
        if A[idx1] == A[idx2]:
            C_even.append(A[idx1])
        else:
            C_even.append(0)

    # Find the maximum length subarray of distinct positive elements in C_odd and C_even
    max_len_odd = solve_sliding_window(C_odd)
    max_len_even = solve_sliding_window(C_even)
    
    # A subarray of length `k` in C corresponds to a 1122 sequence of length `2k` in A.
    # We take the maximum length found from considering both odd-starting and even-starting subarrays.
    print(2 * max(max_len_odd, max_len_even))

# Execute the solve function to run the program
solve()