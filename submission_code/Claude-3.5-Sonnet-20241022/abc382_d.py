def generate_sequences(N, M):
    sequences = []
    
    def backtrack(curr_seq, pos):
        if pos == N:
            sequences.append(curr_seq[:])
            return
            
        # Get minimum value for current position
        min_val = curr_seq[pos-1] + 10 if pos > 0 else 1
        
        # Get maximum value for current position
        max_val = M if pos == N-1 else M - 10*(N-pos-1)
        
        for val in range(min_val, max_val + 1):
            curr_seq.append(val)
            backtrack(curr_seq, pos + 1)
            curr_seq.pop()
    
    backtrack([], 0)
    return sequences

def main():
    # Read input
    N, M = map(int, input().split())
    
    # Generate all valid sequences
    sequences = generate_sequences(N, M)
    
    # Sort sequences lexicographically (not needed as they're generated in order)
    sequences.sort()
    
    # Print output
    print(len(sequences))
    for seq in sequences:
        print(*seq)

if __name__ == "__main__":
    main()