# YOUR CODE HERE
import sys

# Define helper functions for faster input reading if needed,
# though for N=2e5 standard readline should be fine.
# Example:
# def read_int():
#     return int(sys.stdin.readline())
# def read_ints():
#     return map(int, sys.stdin.readline().split())

def solve():
    """
    Solves the problem based on the described logic.
    Reads input N and array A.
    Calculates the sum of f(l, r) over all 1 <= l <= r <= N.
    Prints the result to standard output.
    """
    N = int(sys.stdin.readline())
    
    # Handle the edge case N=0, although constraints state N >= 1.
    if N == 0:
        print(0)
        return
        
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Dictionary to store the last seen index (1-based) for each value.
    # Key: value from array A, Value: 1-based index of its last occurrence.
    last_pos = {} 
    
    # Initialize the total sum we want to compute.
    total_f_sum = 0

    # Iterate through the array using 1-based indexing for positions.
    # The index 'i' represents the position from 1 to N.
    for i in range(1, N + 1):
        # Get the value at the current position.
        # Note that A is 0-indexed, so we access A[i-1].
        v = A[i-1] 
        
        # Retrieve the 1-based index of the previous occurrence of value 'v'.
        # The .get() method with default value 0 handles the case where 'v' is seen for the first time.
        # A previous position of 0 means it hasn't appeared before index 1.
        p_prev = last_pos.get(v, 0)
        
        # The current position (1-based index) is 'i'.
        p_curr = i
        
        # Calculate the contribution of this specific occurrence A[i-1] at index 'i' to the total sum.
        # This occurrence is the *first* appearance of value 'v' within a subsequence A[l...r]
        # if and only if the left endpoint 'l' is after the previous occurrence (l > p_prev)
        # and the current index 'i' is included (l <= i), and the right endpoint 'r' includes 'i' (i <= r <= N).
        # Combining these conditions gives: p_prev < l <= i and i <= r <= N.
        
        # The number of valid choices for 'l' is i - p_prev (indices from p_prev + 1 to i).
        # The number of valid choices for 'r' is N - i + 1 (indices from i to N).
        
        # The total number of pairs (l, r) for which this A[i-1] is the first occurrence of 'v'
        # is the product of the number of choices for 'l' and 'r'.
        contribution = (p_curr - p_prev) * (N - p_curr + 1)
        
        # Add this contribution to the total sum.
        total_f_sum += contribution
        
        # Update the dictionary with the latest position for value 'v'.
        last_pos[v] = p_curr

    # Print the final computed total sum to standard output.
    print(total_f_sum)

# Execute the solve function to run the program logic.
solve()

# END OF YOUR CODE HERE