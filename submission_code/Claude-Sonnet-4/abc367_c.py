def generate_sequences(n, k, r_values):
    valid_sequences = []
    
    def backtrack(current_sequence, position):
        if position == n:
            if sum(current_sequence) % k == 0:
                valid_sequences.append(current_sequence[:])
            return
        
        for value in range(1, r_values[position] + 1):
            current_sequence.append(value)
            backtrack(current_sequence, position + 1)
            current_sequence.pop()
    
    backtrack([], 0)
    return valid_sequences

# Read input
n, k = map(int, input().split())
r_values = list(map(int, input().split()))

# Generate all valid sequences
sequences = generate_sequences(n, k, r_values)

# Output
for seq in sequences:
    print(' '.join(map(str, seq)))