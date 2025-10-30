def main():
    N, M = map(int, input().split())
    
    sequences = []
    
    def backtrack(seq, pos):
        if pos == N:
            sequences.append(seq[:])
            return
        
        # Determine the range for the current element
        min_val = seq[-1] + 10 if pos > 0 else 1
        max_val = M - 10 * (N - pos - 1)
        
        for val in range(min_val, max_val + 1):
            seq.append(val)
            backtrack(seq, pos + 1)
            seq.pop()
    
    backtrack([], 0)
    
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()