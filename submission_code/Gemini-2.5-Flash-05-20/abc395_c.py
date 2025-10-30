# YOUR CODE HERE
import sys

def solve():
    # Read N, the number of elements in the array
    N = int(sys.stdin.readline())

    # Read the array A elements.
    # sys.stdin.readline().split() reads the line and splits it by whitespace.
    # map(int, ...) converts each resulting string to an integer.
    # list(...) converts the map object into a list.
    A = list(map(int, sys.stdin.readline().split()))

    # last_seen dictionary will store the most recent index for each value encountered.
    # Key: value from array A (e.g., 3, 9, 5)
    # Value: the 0-based index where that value was last seen (e.g., 0, 1, 2)
    last_seen = {}

    # Initialize min_length to positive infinity.
    # This ensures that any valid length found will be smaller and update this value.
    min_length = float('inf')

    # Iterate through the array A using its index 'i'
    for i in range(N):
        current_value = A[i]

        # Check if the current_value has been seen before.
        # This means there's a previous occurrence of this value.
        if current_value in last_seen:
            # Get the index of its previous occurrence.
            prev_index = last_seen[current_value]

            # Calculate the length of the subarray that contains these two occurrences.
            # The subarray starts at prev_index and ends at current index i.
            # Its length is (current_index - prev_index + 1).
            current_length = i - prev_index + 1

            # Update min_length if the current_length is shorter than the minimum found so far.
            min_length = min(min_length, current_length)
        
        # Always update the last seen index for the current_value to the current index 'i'.
        # This is crucial for finding the *shortest* subarray.
        # If a value appears multiple times, we only care about the distance to its immediately
        # preceding occurrence to find the shortest possible repeating segment.
        last_seen[current_value] = i

    # After iterating through the entire array, check the result.
    # If min_length is still infinity, it means no repeated values were found at all.
    if min_length == float('inf'):
        print("-1")
    else:
        # Otherwise, print the shortest length found.
        print(min_length)

# Call the solve function to execute the program logic.
solve()