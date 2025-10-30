# YOUR CODE HERE
import sys

def solve():
    # Read N from standard input
    n = int(sys.stdin.readline())
    # Read the sequence A from standard input
    # sys.stdin.readline().split() reads the line and splits it into strings based on whitespace.
    # map(int, ...) converts each string token into an integer.
    # list(...) converts the map object into a list.
    a = list(map(int, sys.stdin.readline().split()))

    # Dictionary to store the most recent 1-based index seen for each value in A.
    # Keys will be the values from A (A_i).
    # Values will be the 1-based index (position) where that value was last encountered.
    last_pos = {}

    # List to store the resulting sequence B.
    # We pre-allocate a list of size N with placeholder values (e.g., 0)
    # This allows direct assignment using index `i` inside the loop.
    result_b = [0] * n

    # Iterate through the input sequence A using a 0-based index `i`.
    # The loop runs from i = 0 to N-1.
    for i in range(n):
        # Get the current value from the sequence A at index `i`.
        current_value = a[i]
        
        # Calculate the corresponding 1-based index for the current position.
        # The problem statement uses 1-based indexing for positions.
        current_index_1_based = i + 1

        # Check if the `current_value` has been seen before by looking it up in the `last_pos` dictionary.
        if current_value in last_pos:
            # If `current_value` is found in `last_pos`, it means we have encountered this value previously.
            # The value stored in `last_pos[current_value]` is the 1-based index of its most recent previous occurrence.
            # This is exactly what B_i should be according to the problem definition.
            # Assign this index to the `i`-th position in the result list `result_b`.
            result_b[i] = last_pos[current_value]
        else:
            # If `current_value` is not found in `last_pos`, it means this is the first time we are seeing this value
            # during our iteration, or more precisely, there is no occurrence of this value at an index `j < i`.
            # According to the problem definition, in this case, B_i should be -1.
            # Assign -1 to the `i`-th position in the result list `result_b`.
            result_b[i] = -1

        # After determining B_i (which is `result_b[i]`), we must update the `last_pos` dictionary.
        # We record the current 1-based index (`current_index_1_based`) as the most recent position
        # where `current_value` was found. This ensures that if we encounter `current_value` again later
        # in the sequence, we will correctly report `current_index_1_based` as its preceding position.
        last_pos[current_value] = current_index_1_based

    # After iterating through the entire sequence A, the `result_b` list contains all the required values B_1, ..., B_N.
    # We need to print these values in one line, separated by spaces.
    # `map(str, result_b)` converts each integer element in `result_b` into its string representation.
    # `' '.join(...)` concatenates these strings into a single string, with a space character used as the separator.
    # `print(...)` outputs this final string to standard output.
    print(' '.join(map(str, result_b)))

# Call the solve function to execute the logic
solve()