# YOUR CODE HERE
import sys

# Read N and M from standard input
N, M = map(int, sys.stdin.readline().split())

results = []

def generate_sequences(k, current_sequence):
    """
    Recursively generates integer sequences A_1, ..., A_N satisfying the conditions.

    Args:
        k: The index of the element we are determining (1-based: 1 to N).
        current_sequence: The list of elements A_1, ..., A_{k-1} determined so far.
    """
    # Access the list storing results in the outer scope
    global results

    if k == N + 1:
        # Base case: Sequence A_1, ..., A_N is complete
        # The constraints applied during generation ensure the sequence is valid.
        results.append(list(current_sequence)) # Add a copy of the sequence
        return

    if k == 1:
        # Determining A_1
        # Conditions: 1 <= A_1 and A_N <= M and A_N >= A_1 + 10*(N-1)
        # From A_N <= M and A_N >= A_1 + 10*(N-1), we get A_1 <= M - 10*(N-1).
        # Combined with 1 <= A_1, the range for A_1 is [1, M - 10*(N-1)].
        min_val = 1
        max_val = M - 10 * (N - 1)

        # Iterate through possible values for A_1 in increasing order for lexicographical output
        # The range function is exclusive of the stop value, so we add 1 to max_val.
        # If min_val > max_val, range(min_val, max_val + 1) is empty, which is correct.
        for val in range(min_val, max_val + 1):
            # Start the sequence with the chosen value for A_1
            generate_sequences(k + 1, [val])

    else:
        # Determining A_k (where k > 1)
        # Conditions: A_k >= A_{k-1} + 10 and A_N <= M and A_N >= A_k + 10*(N-k)
        # Let last_val = A_{k-1} (the last element in current_sequence).
        # From A_k >= A_{k-1} + 10, the minimum value for A_k is last_val + 10.
        # From A_N <= M and A_N >= A_k + 10*(N-k), we get A_k <= M - 10*(N-k).
        # Combined, the range for A_k is [last_val + 10, M - 10*(N-k)].
        last_val = current_sequence[-1] # A_{k-1}

        min_val = last_val + 10
        max_val = M - 10 * (N - k)

        # Iterate through possible values for A_k in increasing order
        # The range function is exclusive of the stop value, so we add 1 to max_val.
        # If min_val > max_val, range(min_val, max_val + 1) is empty, which is correct.
        for val in range(min_val, max_val + 1):
             # Append the current value `val` to the sequence and proceed to determine A_{k+1}.
             # We create a new list `current_sequence + [val]` to avoid modifying the list
             # shared by parallel branches in the recursion tree.
             generate_sequences(k + 1, current_sequence + [val])


# Start the generation process from the first element (index k=1) with an empty initial sequence.
generate_sequences(1, [])

# Print the total count of sequences found
print(len(results))

# Print each found sequence
for seq in results:
    # Print elements of the sequence separated by spaces using the unpack operator (*)
    print(*seq)