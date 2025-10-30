# YOUR CODE HERE
def generate_sequences(N, M):
    def backtrack(sequence, index):
        if index == N:
            sequences.append(sequence[:])
            return
        
        start = sequence[-1] + 10 if index > 0 else 1
        end = min(M, M - 10 * (N - index - 1))
        
        for num in range(start, end + 1):
            sequence.append(num)
            backtrack(sequence, index + 1)
            sequence.pop()

    sequences = []
    backtrack([], 0)
    return sequences

# Read input
N, M = map(int, input().split())

# Generate sequences
result = generate_sequences(N, M)

# Print output
print(len(result))
for seq in result:
    print(' '.join(map(str, seq)))