MOD = 998244353

def count_valid_sequences(N, A, B):
    from itertools import permutations

    # Generate all possible sequences of length 3 using integers 1 to N
    all_sequences = list(permutations(range(1, N + 1), 3))

    # Sort all sequences and their reverses
    sorted_sequences = sorted(all_sequences)
    sorted_reverses = sorted([tuple(reversed(seq)) for seq in all_sequences])

    # Create a mapping from sequences to their positions
    seq_to_pos = {seq: i + 1 for i, seq in enumerate(sorted_sequences)}
    rev_to_pos = {rev: i + 1 for i, rev in enumerate(sorted_reverses)}

    # Function to check if a given set of sequences satisfies the conditions
    def is_valid(sequences):
        a = [0] * N
        b = [0] * N
        for i, seq in enumerate(sequences):
            a[i] = seq_to_pos[seq]
            b[i] = rev_to_pos[tuple(reversed(seq))]

        # Check if a and b match the given A and B
        for i in range(N):
            if a[i] != A[i]:
                return False
            if B[i] != -1 and b[i] != B[i]:
                return False

        # Check if all values in a and b are unique
        if len(set(a)) != N or len(set(b)) != N:
            return False

        return True

    # Count the number of valid sets of sequences
    count = 0
    for sequences in permutations(all_sequences, N):
        if is_valid(sequences):
            count += 1
            count %= MOD

    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:]))

# Calculate and print the result
result = count_valid_sequences(N, A, B)
print(result)