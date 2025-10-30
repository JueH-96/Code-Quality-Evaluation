def generate_sequences(n, k, r_values):
    valid_sequences = []
    
    def backtrack(index, current_sequence):
        if index == n:
            # Check if sum is divisible by K
            if sum(current_sequence) % k == 0:
                valid_sequences.append(current_sequence[:])
            return
        
        # Try all possible values for current position
        for value in range(1, r_values[index] + 1):
            current_sequence.append(value)
            backtrack(index + 1, current_sequence)
            current_sequence.pop()
    
    backtrack(0, [])
    return valid_sequences

# Read input
n, k = map(int, input().split())
r_values = list(map(int, input().split()))

# Generate and print sequences
sequences = generate_sequences(n, k, r_values)
for seq in sequences:
    print(' '.join(map(str, seq)))