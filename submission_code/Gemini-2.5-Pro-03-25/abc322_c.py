# YOUR CODE HERE
import sys

def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the list of firework days A from standard input
    # The list A contains M integers: A_1, A_2, ..., A_M
    # These are 1-based day numbers.
    # The list is guaranteed to be sorted: A_1 < A_2 < ... < A_M
    # It is also guaranteed that A_M = N.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize a pointer `ptr` to 0. This pointer will keep track of the index
    # in the list A corresponding to the next upcoming firework day.
    ptr = 0
    
    # List to store the results for each day i from 1 to N.
    # Storing results as strings allows for efficient joining later.
    results = []

    # Iterate through each day `i` from 1 to N.
    for i in range(1, N + 1):
        # We need to find the smallest firework day `A[k]` such that `A[k] >= i`.
        # Since the list A is sorted, we can maintain a pointer `ptr` to the index
        # of the current target firework day.
        # As `i` increases, the target firework day might change.
        # Specifically, if the current target firework day `A[ptr]` is before day `i`,
        # we need to advance the pointer `ptr` to find the next firework day.
        # We advance `ptr` as long as `A[ptr]` is strictly less than `i`.
        # This ensures that when the loop stops, `A[ptr]` is the first element in A
        # that is greater than or equal to `i`.
        
        # The condition `A[ptr] < i` checks if the firework day at index `ptr` has passed.
        # Since `A` is sorted and `A[M-1] = N`, for any `i <= N`, there will always be
        # a firework day `A[k]` such that `A[k] >= i`. This guarantees that `ptr`
        # will not go out of bounds (i.e., `ptr` will always be less than `M`).
        while A[ptr] < i:
            ptr += 1
            # The maximum value ptr can reach is M-1, which corresponds to A_M = N.
        
        # After the while loop, A[ptr] is the date of the first fireworks display
        # occurring on or after day i.
        # The problem asks for the number of days *later* this occurs.
        # If fireworks are on day i, it's 0 days later.
        # If the next fireworks are on day `A[ptr]`, it's `A[ptr] - i` days later.
        days_later = A[ptr] - i
        
        # Append the calculated result (as a string) to the results list.
        results.append(str(days_later))

    # After iterating through all days from 1 to N, print all the results.
    # Each result should be on a new line. Joining the list of strings with
    # newline characters achieves this efficiently.
    print("
".join(results))

# Execute the solve function to run the program.
solve()