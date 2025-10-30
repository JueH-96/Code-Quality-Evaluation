import sys
import itertools

MOD = 998244353

def count_valid_sequences(N, A, B):
    # Generate all permutations of 1 through N
    perms = list(itertools.permutations(range(1, N + 1)))

    # Generate all possible sequences of length 3
    sequences = []
    for perm in perms:
        for p in itertools.permutations(perm):
            sequences.append(p)

    # Define the sequences a and b
    def get_rank(seq):
        rank = 1
        for s in sequences:
            if seq > s:
                rank += 1
        return rank

    def get_reverse_rank(seq):
        rank = 1
        for s in sequences:
            if seq[::-1] > s:
                rank += 1
        return rank

    valid_count = 0
    for seq in sequences:
        a = tuple(get_rank(seq[i:i+1]) for i in range(N))
        b = tuple(get_reverse_rank(seq[i:i+1]) for i in range(N))

        if all(a[i] == A[i] for i in range(N)) and all(b[i] == B[i] if B[i] != -1 else True for i in range(N)):
            valid_count += 1

    return valid_count % MOD

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:]))

# Calculate and print the result
result = count_valid_sequences(N, A, B)
print(result)