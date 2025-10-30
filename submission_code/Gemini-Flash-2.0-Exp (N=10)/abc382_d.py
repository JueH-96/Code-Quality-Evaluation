def solve():
    n, m = map(int, input().split())
    
    sequences = []
    
    def generate_sequences(current_sequence, start_val):
        if len(current_sequence) == n:
            sequences.append(current_sequence.copy())
            return
        
        if not current_sequence:
            for i in range(start_val, m + 1):
                generate_sequences([i], i + 10)
        else:
            last_val = current_sequence[-1]
            for i in range(last_val + 10, m + 1):
                if len(current_sequence) == n - 1 and i <= m:
                    generate_sequences(current_sequence + [i], i + 10)
                elif len(current_sequence) < n - 1:
                    generate_sequences(current_sequence + [i], i + 10)
    
    generate_sequences([], 1)
    
    valid_sequences = []
    for seq in sequences:
        if seq[-1] <= m:
            valid_sequences.append(seq)

    print(len(valid_sequences))
    for seq in valid_sequences:
        print(*seq)

solve()