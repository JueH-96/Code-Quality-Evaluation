import sys

# Global variables to store N, M, and the list of all valid sequences.
# Using globals simplifies passing them around in the recursive function,
# especially N and M which are constant once read.
N = 0
M = 0
all_sequences = []

def generate_sequences(idx, current_sequence):
    """
    Recursively generates sequences (A_1, ..., A_N) satisfying the conditions.

    Args:
        idx (int): The current 1-based index (A_idx) of the element to determine.
        current_sequence (list): The list containing elements A_1, ..., A_{idx-1}
                                 built so far. This list is modified in place
                                 (append/pop) for efficiency and backtracking.
    """
    global N, M, all_sequences

    # Base case: If idx is N + 1, it means we have successfully determined
    # A_1 through A_N, forming a complete sequence.
    if idx == N + 1:
        # Append a copy of the sequence. We use list(current_sequence) because
        # current_sequence is modified by subsequent pop() operations.
        all_sequences.append(list(current_sequence))
        return

    # Determine the minimum possible value for A_idx.
    # Condition 1: A_i >= 1
    min_val = 1
    if idx > 1:
        # Condition 2: A_i >= A_{i-1} + 10
        # A_{i-1} is the last element added to current_sequence, which is at index -1.
        min_val = current_sequence[-1] + 10

    # Determine the maximum possible value for A_idx.
    # Condition 3: A_N <= M
    # Also, A_N must be at least A_idx + (N - idx) * 10 (by applying condition 2 repeatedly).
    # So, A_idx + (N - idx) * 10 <= M
    # This implies: A_idx <= M - (N - idx) * 10
    max_val = M - (N - idx) * 10

    # Iterate through all valid integer values for A_idx in increasing order.
    # This crucial step ensures that sequences are generated in lexicographical order.
    for val in range(min_val, max_val + 1):
        # Add the current value to the sequence.
        current_sequence.append(val)
        # Recurse: Call the function to determine the next element A_{idx+1}.
        generate_sequences(idx + 1, current_sequence)
        # Backtrack: Remove the current value from the sequence. This allows
        # the loop to try the next possible value for A_idx without interference,
        # and restores the current_sequence state for parent calls.
        current_sequence.pop()

def solve():
    """
    Reads input, initiates the sequence generation, and prints the results.
    """
    global N, M

    # Read N and M from standard input.
    N, M = map(int, sys.stdin.readline().split())

    # Start the recursive generation process from the first element (idx=1)
    # with an initially empty sequence list.
    generate_sequences(1, [])

    # Print the total number of sequences found.
    print(len(all_sequences))
    # Print each sequence. The *seq unpacks the list elements as separate arguments
    # to print(), which then prints them separated by spaces by default.
    for seq in all_sequences:
        print(*seq)

# Call the solve function to execute the program.
solve()