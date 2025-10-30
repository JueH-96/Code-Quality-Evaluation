# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the array A
    # Use 0-based indexing internally
    A = list(map(int, sys.stdin.readline().split()))

    # Dictionary to store the last seen index (0-based) of each element value
    # Key: element value, Value: latest index where this value was encountered
    last_seen = {}
    
    # Initialize minimum length to a value larger than any possible length
    # A subarray of length 1 cannot have duplicates. Minimum possible length is 2.
    # The maximum possible length is N.
    # Using float('inf') works well as an initial large value.
    min_length = float('inf')

    # Iterate through the array using 0-based index i
    for i in range(N):
        value = A[i]
        
        # Check if the current value has been seen before
        if value in last_seen:
            # If yes, retrieve the index of the most recent previous occurrence
            prev_index = last_seen[value]
            
            # The subarray from prev_index to i (inclusive) is the shortest subarray
            # ending at index i that contains the value 'value' twice.
            # The length of this subarray is i - prev_index + 1.
            current_length = i - prev_index + 1
            
            # Update the minimum length found so far if the current one is smaller
            min_length = min(min_length, current_length)
            
        # Update the last seen index for the current value to the current index i.
        # This is crucial to find the *shortest* subarray involving consecutive
        # occurrences of the same value in the original array sequence.
        last_seen[value] = i

    # After iterating through the entire array, if min_length is still
    # the initial large value (infinity), it means no repeated elements were found
    # in the array A, and thus no contiguous subarray can have a repeated element.
    if min_length == float('inf'):
        # Print -1 as per the problem statement
        print(-1)
    else:
        # Otherwise, print the minimum length found
        print(min_length)

# This standard Python construct ensures that the solve() function is called
# only when the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    solve()