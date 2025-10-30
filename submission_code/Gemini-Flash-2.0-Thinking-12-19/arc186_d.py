def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    MOD = 998244353
    
    dp = [0] * (n + 1)
    dp[0] = 0
    if n >= 1:
        dp[1] = 1
    for m in range(2, n + 1):
        count = 0
        for v in range(1, m):
            def get_partitions(target_sum, num_parts):
                if num_parts == 0:
                    return [[]] if target_sum == 0 else []
                if target_sum < num_parts:
                    return []
                partitions = []
                for first_part in range(1, target_sum - (num_parts - 1) + 1):
                    for rest_partition in get_partitions(target_sum - first_part, num_parts - 1):
                        partitions.append([first_part] + rest_partition)
                return partitions
                
            partitions = get_partitions(m - 1, v)
            current_v_count = 0
            for partition in partitions:
                product = 1
                for length in partition:
                    product = (product * dp[length]) % MOD
                current_v_count = (current_v_count + product) % MOD
            count = (count + current_v_count) % MOD
        dp[m] = count
        
    memo = {}
    
    def get_n_polish_sequences(v, length):
        if length < 0 or v < 0:
            return 0
        if length == 0:
            return 1 if v == 0 else 0
        if (v, length) in memo:
            return memo[(v, length)]
        
        if v == 0:
            return 0
            
        count = 0
        def get_partitions(target_sum, num_parts):
            if num_parts == 0:
                return [[]] if target_sum == 0 else []
            if target_sum < num_parts:
                return []
            partitions = []
            for first_part in range(1, target_sum - (num_parts - 1) + 1):
                for rest_partition in get_partitions(target_sum - first_part, num_parts - 1):
                    partitions.append([first_part] + rest_partition)
            return partitions

        partitions = get_partitions(length, v)
        current_v_count = 0
        for partition in partitions:
            product = 1
            for part_length in partition:
                product = (product * dp[part_length]) % MOD
            current_v_count = (current_v_count + product) % MOD
        
        memo[(v, length)] = current_v_count
        return current_v_count
        
    memo_count = {}
    
    def count_leq_polish(index, is_tight):
        if index > n:
            return 1
        if (index, is_tight) in memo_count:
            return memo_count[(index, is_tight)]
        
        count = 0
        limit = a[index-1] if is_tight else n - 1
        
        for v in range(limit + 1):
            next_is_tight = (is_tight and v == limit)
            if index == 1:
                if v == 0:
                    if n == 1:
                        current_count = 1
                    else:
                        current_count = 0
                elif v >= 1:
                    current_count = get_n_polish_sequences(v, n - 1)
                else:
                    current_count = 0
            else:
                if v >= 1:
                    current_count = get_n_polish_sequences(v, n - index)
                else:
                    current_count = 0
                    
            if current_count > 0:
                count = (count + (current_count * count_leq_polish(index + 1, next_is_tight)) % MOD) % MOD
                
        memo_count[(index, is_tight)] = count
        return count
        
    result = count_leq_polish(1, True)
    print(result)

if __name__ == '__main__':
    solve()