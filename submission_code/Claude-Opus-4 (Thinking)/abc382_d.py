def generate_sequences(n, m, pos, current_seq, all_seqs):
    if pos == n:
        all_seqs.append(current_seq[:])
        return
    
    if pos == 0:
        min_val = 1
    else:
        min_val = current_seq[pos-1] + 10
    
    # Maximum value such that we can still complete the sequence
    # If we choose value v at position pos, then position n-1 needs at least v + 10*(n-1-pos)
    # Since position n-1 must be <= m, we need v + 10*(n-1-pos) <= m
    max_val = m - 10 * (n - 1 - pos)
    
    if min_val > max_val:
        return
    
    for val in range(min_val, max_val + 1):
        current_seq.append(val)
        generate_sequences(n, m, pos + 1, current_seq, all_seqs)
        current_seq.pop()

n, m = map(int, input().split())
all_seqs = []
generate_sequences(n, m, 0, [], all_seqs)

print(len(all_seqs))
for seq in all_seqs:
    print(' '.join(map(str, seq)))