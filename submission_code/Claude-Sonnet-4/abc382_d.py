# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    
    sequences = []
    
    def backtrack(current_seq, pos):
        if pos == N:
            sequences.append(current_seq[:])
            return
        
        # Determine the minimum value for current position
        if pos == 0:
            min_val = 1
        else:
            min_val = current_seq[pos-1] + 10
        
        # Determine the maximum value for current position
        # We need to ensure we can still build a valid sequence
        # The minimum possible value for the last position would be:
        # current_val + 10 * (N - 1 - pos)
        # This must be <= M
        max_val = M - 10 * (N - 1 - pos)
        
        # Try all valid values for current position
        for val in range(min_val, min(max_val + 1, M + 1)):
            current_seq.append(val)
            backtrack(current_seq, pos + 1)
            current_seq.pop()
    
    backtrack([], 0)
    
    # Output results
    print(len(sequences))
    for seq in sequences:
        print(*seq)

solve()