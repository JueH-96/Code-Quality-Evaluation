def solve():
    n, k = map(int, input().split())
    p_input = list(map(int, input().split()))
    p_0_based = [x - 1 for x in p_input]
    
    permutation_history = []
    permutation_history.append(list(p_0_based))
    seen_permutations = {tuple(p_0_based): 0}
    
    current_p = list(p_0_based)
    period = -1
    pre_period_length = -1
    
    for operation_count in range(1, n * n + 2): # Let's try to limit the search for period
        next_p = [0] * n
        for i in range(n):
            next_p[i] = current_p[current_p[i]]
        
        next_p_tuple = tuple(next_p)
        if next_p_tuple in seen_permutations:
            start_index = seen_permutations[next_p_tuple]
            period = operation_count - start_index
            pre_period_length = start_index
            break
        else:
            permutation_history.append(next_p)
            seen_permutations[next_p_tuple] = operation_count
            current_p = next_p
            
    if period == -1:
        result_p_0_based = current_p
    else:
        if k < pre_period_length:
            result_p_0_based = permutation_history[k]
        else:
            index_in_cycle = (k - pre_period_length) % period
            result_index = pre_period_length + index_in_cycle
            result_p_0_based = permutation_history[result_index]
            
    result_p_1_based = [x + 1 for x in result_p_0_based]
    print(*(result_p_1_based))

if __name__ == '__main__':
    solve()