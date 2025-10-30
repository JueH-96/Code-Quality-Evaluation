def generate(seq):
    if len(seq) == N:
        if seq[-1] <= M:
            sequences.append(seq)
    else:
        start = seq[-1] + 10 if seq else 1
        if len(seq) < N - 1:
            end = M - 10 * (N - len(seq) - 1)
        else:
            end = M
        for A_i in range(start, end + 1):
            generate(seq + [A_i])

# Read N and M from input
input_line = input().strip()
N, M = map(int, input_line.split())

# Collect all sequences
sequences = []

# Generate sequences
generate([])

# Print the number of sequences
print(len(sequences))

# Print each sequence
for seq in sequences:
    print(' '.join(map(str, seq)))