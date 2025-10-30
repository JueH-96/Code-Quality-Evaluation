def solve():
    n = int(input())
    a = list(map(int, input().split()))
    memo_polish_count = {}
    
    def get_polish_count(length):
        if length == 0:
            return 0
        if length == 1:
            return 1
        if length in memo_polish_count:
            return memo_polish_count[length]
        count = 0
        for v1 in range(1, length):
            def get_partitions(target_sum, num_parts):
                if num_parts == 0:
                    return [[]] if target_sum == 0 else []
                if target_sum < num_parts:
                    return []
                partitions = []
                for first_part in range(1, target_sum - num_parts + 2):
                    remaining_sum = target_sum - first_part
                    sub_partitions = get_partitions(remaining_sum, num_parts - 1)
                    for part in sub_partitions:
                        partitions.append([first_part] + part)
                return partitions

            partitions = get_partitions(length - 1, v1)
            v1_count = 0
            for partition in partitions:
                product = 1
                for part_length in partition:
                    product = (product * get_polish_count(part_length)) % 998244353
                v1_count = (v1_count + product) % 998244353
            count = (count + v1_count) % 998244353
        memo_polish_count[length] = count
        return count

    polish_counts = {}
    for i in range(1, n + 1):
        polish_counts[i] = get_polish_count(i)

    memo_count_le = {}

    def count_polish_le_recursive(current_index, is_tight):
        if current_index > n:
            return 1
        
        state = (current_index, is_tight)
        if state in memo_count_le:
            return memo_count_le[state]
        
        count = 0
        limit = a[current_index-1] if is_tight else n - 1
        
        for v1 in range(limit + 1):
            if current_index == 1 and v1 == 0:
                if n == 1:
                    next_tight = is_tight and (v1 == a[current_index-1])
                    count = (count + count_polish_le_recursive(current_index + 1, next_tight)) % 998244353
                continue
            if current_index > 1 and v1 == 0:
                continue
            if v1 >= n:
                continue
                
            valid_v1 = True
            if current_index == 1 and n > 1 and v1 == 0:
                valid_v1 = False
            if not valid_v1:
                continue
                
            if v1 > 0:
                def get_partitions(target_sum, num_parts):
                    if num_parts == 0:
                        return [[]] if target_sum == 0 else []
                    if target_sum < num_parts:
                        return []
                    partitions = []
                    for first_part in range(1, target_sum - num_parts + 2):
                        remaining_sum = target_sum - first_part
                        sub_partitions = get_partitions(remaining_sum, num_parts - 1)
                        for part in sub_partitions:
                            partitions.append([first_part] + part)
                    return partitions
                    
                partitions = get_partitions(n - current_index, v1)
                v1_count = 0
                for partition in partitions:
                    product = 1
                    for part_length in partition:
                        product = (product * polish_counts[part_length]) % 998244353
                    v1_count = (v1_count + product) % 998244353
                
                if v1 < a[current_index-1]:
                    count = (count + v1_count) % 998244353
                elif v1 == a[current_index-1]:
                    next_tight = True
                    count = (count + v1_count) % 998244353
                    
            elif v1 == 0:
                if n == 1 and current_index == 1:
                    next_tight = is_tight and (v1 == a[current_index-1])
                    count = (count + count_polish_le_recursive(current_index + 1, next_tight)) % 998244353
                    
        memo_count_le[state] = count
        return count

    if n == 1:
        if a[0] >= 0:
            print(1)
        else:
            print(0)
    else:
        result = 0
        first_val_limit = a[0]
        for v1 in range(1, min(first_val_limit, n - 1) + 1):
            def get_partitions(target_sum, num_parts):
                if num_parts == 0:
                    return [[]] if target_sum == 0 else []
                if target_sum < num_parts:
                    return []
                partitions = []
                for first_part in range(1, target_sum - num_parts + 2):
                    remaining_sum = target_sum - first_part
                    sub_partitions = get_partitions(remaining_sum, num_parts - 1)
                    for part in sub_partitions:
                        partitions.append([first_part] + part)
                return partitions

            partitions = get_partitions(n - 1, v1)
            v1_count = 0
            for partition in partitions:
                product = 1
                for part_length in partition:
                    product = (product * polish_counts[part_length]) % 998244353
                v1_count = (v1_count + product) % 998244353
            result = (result + v1_count) % 998244353

        if a[0] >= 1:
            v1 = a[0]
            if v1 < n:
                initial_tight = True
                result = (result + count_polish_le_recursive(2, initial_tight)) % 998244353

        print(result)

solve()