def find_middle_occurrences(sequence, N):
    # Create a dictionary to store the first and second occurrences
    occurrences = {}
    # List to store the middle occurrences
    middle_occurrences = [0] * N

    # Iterate over the sequence to find the middle occurrences
    for idx, value in enumerate(sequence):
        if value not in occurrences:
            occurrences[value] = [idx]
        else:
            occurrences[value].append(idx)
            if len(occurrences[value]) == 2:
                # Store the index of the middle occurrence
                middle_occurrences[value - 1] = occurrences[value][1]

    # Sort the numbers based on the middle occurrences
    sorted_numbers = sorted(range(1, N + 1), key=lambda i: middle_occurrences[i - 1])

    return sorted_numbers

# Read input from stdin
N = int(input().strip())
sequence = list(map(int, input().strip().split()))

# Find and print the sorted sequence
sorted_sequence = find_middle_occurrences(sequence, N)
print(' '.join(map(str, sorted_sequence)))