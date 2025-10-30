def solve():
    n, m = map(int, input().split())
    
    sequences = []
    
    def find_sequences(current_sequence):
        if len(current_sequence) == n:
            sequences.append(current_sequence)
            return
        
        if not current_sequence:
            start = 1
        else:
            start = current_sequence[-1] + 10
        
        for i in range(start, m + 1):
            if len(current_sequence) == n - 1 and i > m:
                break
            find_sequences(current_sequence + [i])
            
    find_sequences([])
    
    valid_sequences = []
    for seq in sequences:
        if len(seq) == n and seq[-1] <= m:
            valid_sequences.append(seq)
            
    print(len(valid_sequences))
    for seq in sorted(valid_sequences):
        print(*seq)

solve()