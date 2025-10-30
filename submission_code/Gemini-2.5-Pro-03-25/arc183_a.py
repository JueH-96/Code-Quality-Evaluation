# YOUR CODE HERE
import math
import sys

# Set a higher recursion depth if needed, although this solution is iterative.
# sys.setrecursionlimit(2000) 

def solve():
    """
    Solves the problem of finding the floor((S+1)/2)-th lexicographically smallest
    good integer sequence, where S is the total number of such sequences.
    A good sequence of length NK has K occurrences of each integer from 1 to N.
    The algorithm constructs the sequence element by element from left to right.
    """
    N, K = map(int, sys.stdin.readline().split())
    
    total_len = N * K
    
    # Handle the edge case where NK=0. Based on constraints N, K >= 1, total_len >= 1.
    if total_len == 0:
         # An empty sequence is the only possibility. Output format typically requires a newline.
         print() 
         return

    # Initialize counts array for numbers 1 to N. Using 1-based indexing for convenience.
    # counts[i] stores the remaining number of times integer 'i' must appear.
    counts = [0] * (N + 1) 
    for i in range(1, N + 1):
        counts[i] = K 
    
    # Calculate the total number of good sequences S.
    # S is the multinomial coefficient (NK)! / (K!)^N.
    # This can be computed efficiently as a product of binomial coefficients:
    # S = C(NK, K) * C(NK-K, K) * ... * C(K, K)
    current_S_len_for_comb = total_len
    S = 1
    # Use math.comb which handles large integers and is generally efficient.
    for _ in range(N):
        # Check if the length is non-negative before calling math.comb
        # This path should not be reachable with N, K >= 1.
        if current_S_len_for_comb < 0:
             S = 0 # Indicates an error or unexpected state
             break
        
        # Calculate term C(current_len, K)
        # math.comb(n, k) returns 0 if k > n, which is mathematically correct.
        try:
            # Ensure K is non-negative (guaranteed by constraints K >= 1)
            if K < 0:
                 raise ValueError("K cannot be negative")

            term = math.comb(current_S_len_for_comb, K)
            S *= term
            
            # Optimization: if S becomes 0, further terms will also be 0. Stop early.
            # This scenario is unlikely for valid inputs N, K >= 1 where S should be positive.
            if S == 0:
                break 
            
            current_S_len_for_comb -= K
        except ValueError as e:
             # Catch potential errors from math.comb, e.g., negative arguments.
             print(f"ValueError during combination calculation: {e}", file=sys.stderr)
             # Additional debug info could be printed here if needed
             # print(f"State: N={N}, K={K}, current_S_len_for_comb={current_S_len_for_comb}", file=sys.stderr)
             return # Exit on error

    # If S calculation resulted in 0 (unexpected for valid inputs), handle appropriately.
    # For N, K >= 1, S should be at least 1.
    if S == 0:
        # This suggests either an issue or an edge case not covered by constraints.
        # Perhaps print an empty line or specific error message.
        # Assuming S >= 1 based on problem context (e.g., Sample 2).
        pass # Continue assuming S >= 1.

    # Calculate the target rank M = floor((S+1)/2). Python's // operator performs floor division.
    target_rank = (S + 1) // 2
    
    # List to store the elements of the resulting sequence
    ans = []
    
    # Maintain the total count of sequences possible from the current state (denoted as T)
    current_total_count = S 
    
    # Maintain the remaining length of the sequence to construct
    remaining_len = total_len
    
    # Construct the sequence element by element
    for _ in range(total_len):
        
        # If remaining length is 0, the sequence is fully constructed.
        if remaining_len == 0:
             break

        # Iterate through possible numbers (1 to N) for the current position
        for i in range(1, N + 1):
            # Check if number 'i' is available to be placed (its count > 0)
            if counts[i] > 0:
                
                # Calculate the number of sequences that can be formed starting with
                # the current prefix followed by 'i'. This uses the property that
                # the count C_i = T * k_i / L', where T is current_total_count,
                # k_i is counts[i], and L' is remaining_len.
                
                # Safety check to prevent division by zero.
                if remaining_len <= 0:
                    # This state indicates a logical error as remaining_len should be > 0.
                    print(f"Error: Division by zero or negative length attempted. State: N={N}, K={K}, remaining_len={remaining_len}", file=sys.stderr)
                    return 

                # Use integer division //
                count_i = (current_total_count * counts[i]) // remaining_len
                
                # Decide whether to pick number 'i' based on the target rank
                if target_rank <= count_i:
                    # If the target rank falls within the count for 'i', then 'i' is the correct number for this position.
                    ans.append(i)
                    
                    # Update the state for the next step (next position in the sequence):
                    current_total_count = count_i # The new total count for the subproblem is count_i
                    counts[i] -= 1             # Decrement the count of available 'i's
                    remaining_len -= 1         # Decrease the remaining length to fill
                    
                    # Break the inner loop as we've determined the element for the current position
                    break 
                else:
                    # If the target rank is greater than count_i, the desired sequence is lexicographically larger
                    # than any sequence starting with the current prefix + 'i'.
                    # Subtract count_i from the target rank to adjust for the skipped sequences.
                    target_rank -= count_i
                    # Continue the inner loop to check the next available number (i+1)
                    
    # Print the constructed sequence elements separated by spaces
    print(*(ans))

# Execute the solve function when the script is run
solve()