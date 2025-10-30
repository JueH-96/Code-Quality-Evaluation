def solve():
    n, q = map(int, input().split())
    operations = []
    for _ in range(q):
        operations.append(tuple(map(int, input().split())))
    
    dp = {}
    initial_sequence = tuple([0] * n)
    dp[initial_sequence] = 1
    
    mod = 998244353
    
    for i in range(q):
        p_i, v_i = operations[i]
        next_dp = {}
        for current_sequence_tuple, count in dp.items():
            current_sequence = list(current_sequence_tuple)
            
            # Operation 1: Replace S_1, ..., S_{P_i} with V_i
            prefix_segment = current_sequence[:p_i]
            if not any(x > v_i for x in prefix_segment):
                next_sequence_prefix = list(current_sequence)
                for j in range(p_i):
                    next_sequence_prefix[j] = v_i
                next_sequence_tuple_prefix = tuple(next_sequence_prefix)
                next_dp[next_sequence_tuple_prefix] = next_dp.get(next_sequence_tuple_prefix, 0) + count
                next_dp[next_sequence_tuple_prefix] %= mod
                
            # Operation 2: Replace S_{P_i}, ..., S_N with V_i
            suffix_segment = current_sequence[p_i-1:]
            if not any(x > v_i for x in suffix_segment):
                next_sequence_suffix = list(current_sequence)
                for j in range(p_i - 1, n):
                    next_sequence_suffix[j] = v_i
                next_sequence_tuple_suffix = tuple(next_sequence_suffix)
                next_dp[next_sequence_tuple_suffix] = next_dp.get(next_sequence_tuple_suffix, 0) + count
                next_dp[next_sequence_tuple_suffix] %= mod
                
        dp = next_dp
        
    total_count = 0
    for count in dp.values():
        total_count += count
        total_count %= mod
        
    print(total_count)

if __name__ == '__main__':
    solve()