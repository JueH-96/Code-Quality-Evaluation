import sys

# Function to solve the problem
def solve():
    # Read the integer N (length of the array) from standard input.
    N = int(sys.stdin.readline())

    # Read the array A of N integers from standard input.
    # sys.stdin.readline() reads a line, .split() splits it by whitespace
    # into a list of strings, and map(int, ...) converts each string to an integer.
    # list(...) converts the map object into a list.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize min_length to positive infinity.
    # This variable will store the length of the shortest contiguous subarray
    # that contains at least one repeated element.
    # We use float('inf') because any valid length (which must be >= 2) will be smaller.
    # If this value remains infinity after checking the whole array, it means
    # no such subarray was found.
    min_length = float('inf')

    # Use a dictionary to keep track of the most recent index encountered
    # for each distinct number seen so far in the array A.
    # This helps in efficiently calculating the distance between the current
    # occurrence of a number and its previous occurrence.
    # Key: The number from the array A (A_i).
    # Value: The 0-based index in the array A where this number was most recently seen.
    last_seen = {}

    # Iterate through the array A from left to right, using the index `i`.
    # `i` represents the current index being processed (0-based).
    for i in range(N):
        # Get the value of the element at the current index `i`.
        current_value = A[i]

        # Check if the `current_value` exists as a key in the `last_seen` dictionary.
        # If it does, it means this number has appeared at least once before
        # at the index stored in `last_seen[current_value]`.
        if current_value in last_seen:
            # If the number has been seen before, get the index of its last occurrence.
            # This `last_index` is the starting index of the shortest subarray
            # ending at `i` that contains two occurrences of `current_value`.
            last_index = last_seen[current_value]

            # The contiguous subarray containing the previous occurrence (at `last_index`)
            # and the current occurrence (at `i`) is A[last_index...i].
            # The length of this subarray is calculated as:
            # (index of current occurrence) - (index of last occurrence) + 1
            # Example: indices 2 and 5 -> length 5 - 2 + 1 = 4.
            current_length = i - last_index + 1

            # We are looking for the *shortest* such subarray across the entire array A.
            # Compare the length of the current subarray (`current_length`) with
            # the minimum length found so far (`min_length`), and update `min_length`
            # if `current_length` is smaller.
            min_length = min(min_length, current_length)

        # After processing the current element A[i], update its entry in the
        # `last_seen` dictionary. The current index `i` is now the most recent
        # index where `current_value` was found.
        # This ensures that the next time `current_value` is encountered,
        # `last_seen[current_value]` will provide the index of the occurrence
        # immediately preceding the next one, allowing us to calculate the
        # shortest distance between consecutive occurrences.
        last_seen[current_value] = i

    # After the loop finishes, we have iterated through the entire array.
    # `min_length` now holds the length of the shortest contiguous subarray
    # with a repeated element, or it is still `float('inf')` if no such subarray exists.

    # Check if `min_length` remained at its initial infinite value.
    if min_length == float('inf'):
        # If it's still infinity, it means no number was repeated in the array A.
        # Therefore, no contiguous subarray can contain a repeated element.
        # Print -1 as required by the problem statement.
        print(-1)
    else:
        # If `min_length` was updated, it means a repeated element was found
        # and `min_length` holds the length of the shortest subarray containing one.
        # Print the minimum length found.
        print(min_length)

# Execute the solve function to run the program.
solve()