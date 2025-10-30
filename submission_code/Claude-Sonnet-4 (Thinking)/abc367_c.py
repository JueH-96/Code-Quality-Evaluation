N, K = map(int, input().split())
R = list(map(int, input().split()))

valid_sequences = []

def generate_sequences(sequence, index):
    if index == N:
        if sum(sequence) % K == 0:
            valid_sequences.append(sequence[:])
        return
    
    for val in range(1, R[index] + 1):
        sequence.append(val)
        generate_sequences(sequence, index + 1)
        sequence.pop()

generate_sequences([], 0)
valid_sequences.sort()

for seq in valid_sequences:
    print(' '.join(map(str, seq)))