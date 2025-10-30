def solve():
    n, m = map(int, input().split())
    
    sequences = []
    
    def find_sequences(current_sequence):
        if len(current_sequence) == n:
            if current_sequence[-1] <= m:
                sequences.append(current_sequence)
            return

        last_val = current_sequence[-1] if current_sequence else 0
        for i in range(last_val + 1, m + 1):
            valid = True
            if current_sequence:
                if i < last_val + 10:
                    valid = False
            if valid:
                find_sequences(current_sequence + [i])

    find_sequences([])
    
    print(len(sequences))
    for seq in sequences:
        print(*seq)

solve()