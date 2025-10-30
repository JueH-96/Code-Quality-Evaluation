def solve(L, R):
    # Initialize the list to store the sequences
    sequences = []

    # While L is less than R
    while L < R:
        # Find the maximum power of 2 that is less than or equal to R and is greater than L
        i = 0
        while 2 ** (i + 1) <= R and 2 ** (i + 1) <= L:
            i += 1

        # Find the minimum power of 2 that is greater than L
        j = 0
        while 2 ** j < L:
            j += 1

        # Add the sequence to the list
        sequences.append((2 ** i * j, 2 ** i * (j + 1)))

        # Update L and R
        L = 2 ** i * (j + 1)
        R = R

    # Sort the sequences
    sequences.sort()

    # Print the number of sequences
    print(len(sequences))

    # Print the sequences
    for sequence in sequences:
        print(sequence[0], sequence[1])

# Read the inputs
L, R = map(int, input().split())

# Solve the problem
solve(L, R)