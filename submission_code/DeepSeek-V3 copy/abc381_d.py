# YOUR CODE HERE
def max_1122_subarray_length(N, A):
    from collections import defaultdict

    # To keep track of the last occurrence of each element
    last_occurrence = {}
    # To keep track of the current sequence
    current_sequence = []
    max_length = 0

    for i in range(N):
        current_element = A[i]
        if current_element in last_occurrence:
            # Check if the previous occurrence is part of the current sequence
            prev_index = last_occurrence[current_element]
            if prev_index >= len(current_sequence) - 1:
                # The previous occurrence is not in the current sequence
                # So we can start a new sequence
                current_sequence = []
            else:
                # The previous occurrence is in the current sequence
                # So we can extend the current sequence
                current_sequence.append(current_element)
                # Check if the sequence is valid
                if len(current_sequence) % 2 == 0:
                    # Check if the last two elements are the same
                    if current_sequence[-2] == current_sequence[-1]:
                        # The sequence is valid
                        if len(current_sequence) > max_length:
                            max_length = len(current_sequence)
                    else:
                        # The sequence is invalid, reset
                        current_sequence = []
        else:
            # First occurrence, add to the sequence
            current_sequence.append(current_element)
        # Update the last occurrence
        last_occurrence[current_element] = i

    return max_length

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the result
result = max_1122_subarray_length(N, A)

# Print the result
print(result)