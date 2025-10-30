# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep recursive calls,
# although N<=12 might be handled by the default limit in most environments.
# Uncomment the following line if you encounter RecursionError.
# sys.setrecursionlimit(2000)

def solve():
    """
    Reads input N and M, finds all integer sequences (A_1, ..., A_N)
    satisfying the given conditions, and prints the count followed by
    the sequences in lexicographical order.

    Conditions:
    1. 1 <= A_i
    2. A_{i-1} + 10 <= A_i for i = 2 through N
    3. A_N <= M
    """
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())

    # List to store the resulting valid sequences
    results = []

    def find_sequences_indexed(current_index, current_sequence):
        """
        Recursive helper function to build sequences element by element.

        Args:
            current_index: The 1-based index of the element (A_current_index)
                           that we are currently trying to determine.
            current_sequence: The list containing the elements determined so far
                              (A_1, ..., A_{current_index-1}).
        """
        # Base case: If we have successfully determined all N elements
        if current_index == N + 1:
            # Add a copy of the completed sequence to the results list
            results.append(list(current_sequence))
            return

        # --- Determine the range of possible values for A_{current_index} ---

        # Calculate the lower bound (start_val) for A_{current_index}
        if current_index == 1:
            # For the first element (A_1), the minimum value is 1.
            start_val = 1
        else:
            # For subsequent elements (A_i, i > 1), it must be at least
            # the previous element plus 10 (A_i >= A_{i-1} + 10).
            previous_element = current_sequence[-1]
            start_val = previous_element + 10

        # Calculate the upper bound (max_val) for A_{current_index}.
        # This bound ensures that it's possible to complete the sequence
        # such that the final element A_N does not exceed M.
        # The minimum possible value for A_N, given A_{current_index} = val,
        # occurs when subsequent elements increase by exactly 10 at each step.
        # This minimum A_N is val + (N - current_index) * 10.
        # We require this minimum A_N to be less than or equal to M:
        #   val + (N - current_index) * 10 <= M
        # Rearranging gives the upper bound for val (A_{current_index}):
        #   val <= M - (N - current_index) * 10
        max_val = M - (N - current_index) * 10

        # --- Iterate through valid values and recurse ---

        # Iterate through all possible integer values for A_{current_index}
        # from start_val up to max_val (inclusive).
        # The loop implicitly handles the condition 1 <= A_i because
        # start_val >= 1 always holds (start_val is 1 or previous_element + 10).
        for val in range(start_val, max_val + 1):
            # Choose the value 'val' for the current element A_{current_index}
            current_sequence.append(val)

            # Recursively call the function to determine the next element (A_{current_index + 1})
            find_sequences_indexed(current_index + 1, current_sequence)

            # Backtrack: Remove the last added element ('val') so that we can
            # try the next possible value in the loop for A_{current_index}.
            current_sequence.pop()

    # --- Start the process and print results ---

    # Initial call to the recursive function to start building the sequence
    # from the first element (index 1) with an empty initial sequence.
    find_sequences_indexed(1, [])

    # Print the total number of valid sequences found.
    print(len(results))

    # Print each valid sequence found. The recursive generation process
    # naturally produces them in lexicographical order because we try smaller
    # values for each element before larger ones.
    for seq in results:
        # Print the elements of the sequence separated by spaces.
        print(*seq)

# Execute the solve function to run the main logic of the program.
solve()