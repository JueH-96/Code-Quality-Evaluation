N, M = map(int, input().split())

sequences = []

def backtrack(current_sequence, position):
    if position == N:
        sequences.append(current_sequence[:])
        return
    
    if position == 0:
        min_val = 1
    else:
        min_val = current_sequence[position - 1] + 10
    
    remaining_positions = N - position - 1
    max_val_for_position = M - remaining_positions * 10
    
    for val in range(min_val, max_val_for_position + 1):
        current_sequence.append(val)
        backtrack(current_sequence, position + 1)
        current_sequence.pop()

backtrack([], 0)

print(len(sequences))
for seq in sequences:
    print(' '.join(map(str, seq)))