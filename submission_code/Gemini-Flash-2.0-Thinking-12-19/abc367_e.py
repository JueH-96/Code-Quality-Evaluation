def solve():
    n, k = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    if k == 0:
        print(*(a))
        return
        
    sequences_history = [tuple(a)]
    seen_sequences = {tuple(a): 0}
    
    cycle_start_index = -1
    cycle_length = -1
    
    for i in range(1, n + 2): # Iterate enough times to find a cycle, max cycle length should be at most number of states.
        next_a = [0] * n
        for j in range(n):
            next_a[j] = sequences_history[-1][x[j]-1]
        
        next_a_tuple = tuple(next_a)
        if next_a_tuple in seen_sequences:
            cycle_start_index = seen_sequences[next_a_tuple]
            cycle_length = i - cycle_start_index
            break
        else:
            sequences_history.append(next_a_tuple)
            seen_sequences[next_a_tuple] = i
            
    if cycle_start_index == -1:
        result_sequence = sequences_history[k]
    else:
        pre_cycle_length = cycle_start_index
        if k < pre_cycle_length:
            result_sequence = sequences_history[k]
        else:
            effective_k_in_cycle = (k - pre_cycle_length) % cycle_length
            result_sequence = sequences_history[pre_cycle_length + effective_k_in_cycle]
            
    print(*(list(result_sequence)))

if __name__ == '__main__':
    solve()