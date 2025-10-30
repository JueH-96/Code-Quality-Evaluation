def find_sequences(n, m):
    sequences = []
    
    def generate(current_seq, pos):
        if pos == n:
            sequences.append(current_seq[:])
            return
        
        if pos == 0:
            # First element can be from 1 to m - 10*(n-1)
            # This ensures we can still fit n-1 more elements with gaps of 10
            max_first = m - 10 * (n - 1)
            for val in range(1, max_first + 1):
                current_seq.append(val)
                generate(current_seq, pos + 1)
                current_seq.pop()
        else:
            # For other positions, start from previous + 10
            start = current_seq[-1] + 10
            # For the last position, we can go up to m
            # For other positions, we need to leave room for remaining elements
            if pos == n - 1:
                end = m
            else:
                # Need to leave room for (n - pos - 1) more elements
                # Each needs a gap of at least 10
                end = m - 10 * (n - pos - 1)
            
            for val in range(start, end + 1):
                current_seq.append(val)
                generate(current_seq, pos + 1)
                current_seq.pop()
    
    generate([], 0)
    return sequences

# Read input
n, m = map(int, input().split())

# Find all valid sequences
sequences = find_sequences(n, m)

# Output
print(len(sequences))
for seq in sequences:
    print(' '.join(map(str, seq)))