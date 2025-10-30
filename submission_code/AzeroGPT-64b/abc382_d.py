import itertools

def solution(n, m):
    """
    Generates all integer sequences (A_1, A_2, ..., A_N) of length N that satisfy
    the given conditions and prints them in lexicographical order along with
    the count of such sequences.
    """
    def is_valid(sequence):
        """
        Checks if the sequence meets the specified conditions.
        """
        for i in range(1, len(sequence)):
            if sequence[i] - sequence[i - 1] < 10:
                return False
        return True

    # Generate all possible sequences of length N
    all_sequences = itertools.product(range(1, m + 1), repeat=n)

    # Filter sequences that satisfy the conditions
    valid_sequences = filter(is_valid, all_sequences)

    # Convert to a list to allow indexing
    valid_sequences = list(map(list, valid_sequences))

    # Output the count and sequences
    print(len(valid_sequences))
    for seq in valid_sequences:
        print(' '.join(map(str, seq)))

# Read input values
N, M = map(int, input().split())

# Solve and print the result
solution(N, M)