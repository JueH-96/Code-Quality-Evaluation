def apply_operation(seq):
    """Apply one operation to a sequence."""
    n = len(seq)
    new_seq = [0] * n
    cumsum = 0
    for i in range(n):
        cumsum = (cumsum + seq[i]) % 2
        new_seq[i] = cumsum
    return new_seq

def find_cycle_info(seq, M):
    """Find cycle information for a sequence."""
    seen = {}
    current = seq[:]
    step = 0
    
    while True:
        key = tuple(current)
        if key in seen:
            return seen, seen[key], step - seen[key]
        seen[key] = step
        current = apply_operation(current)
        step += 1
        if step > 2 * M:  # Safety check
            break
    return seen, -1, -1

def solve():
    N, M = map(int, input().split())
    sequences = []
    for _ in range(N):
        sequences.append(list(map(int, input().split())))
    
    MOD = 998244353
    total = 0
    
    # For each sequence, compute its trajectory
    trajectories = []
    for i in range(N):
        seen, cycle_start, cycle_len = find_cycle_info(sequences[i], M)
        trajectories.append((seen, cycle_start, cycle_len))
    
    # Compute f(i, j) for all pairs
    for i in range(N):
        for j in range(i, N):
            if i == j:
                # f(i, i) = 0 (already identical)
                continue
            
            # Find when sequences i and j become identical
            seen_i, start_i, len_i = trajectories[i]
            seen_j, start_j, len_j = trajectories[j]
            
            # Check each step for sequence i
            found = False
            for step_i, state_i in seen_i.items():
                # Check if sequence j reaches this state
                for step_j, state_j in seen_j.items():
                    if state_i == state_j:
                        # They meet at this state
                        # We need the minimum x such that after x operations they're identical
                        if step_i == step_j:
                            total = (total + step_i) % MOD
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # Check if they can meet considering cycles
                max_steps = max(max(seen_i.keys()), max(seen_j.keys())) + max(len_i, len_j)
                seq_i = sequences[i][:]
                seq_j = sequences[j][:]
                
                for x in range(max_steps + 1):
                    if seq_i == seq_j:
                        total = (total + x) % MOD
                        break
                    seq_i = apply_operation(seq_i)
                    seq_j = apply_operation(seq_j)
    
    print(total)

solve()