class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        n = int(s, 2)
        if n <= 1:
            return 0
        
        def count_set_bits_dp(limit_str, target_set_bits):
            memo = {}
            def solve(index, tight, current_set_bits):
                if current_set_bits > target_set_bits:
                    return 0
                if index == len(limit_str):
                    return 1 if current_set_bits == target_set_bits else 0
                if (index, tight, current_set_bits) in memo:
                    return memo[(index, tight, current_set_bits)]
                
                count = 0
                limit = int(limit_str[index]) if tight else 1
                for digit in range(limit + 1):
                    next_tight = tight and (digit == limit)
                    next_set_bits = current_set_bits + digit
                    count += solve(index + 1, next_tight, next_set_bits)
                memo[(index, tight, current_set_bits)] = count
                return count
                
            return solve(0, True, 0)

        def get_count_powers_of_2(limit):
            count = 0
            power_of_2 = 2
            while power_of_2 < limit:
                count += 1
                if power_of_2 > limit // 2:
                    break
                power_of_2 *= 2
            return count
            
        count1 = get_count_powers_of_2(n)
        
        count2 = 0
        max_set_bits_value_index = 0
        max_bits_length = len(s)
        while (1 << (max_set_bits_value_index + 1)) <= max_bits_length:
            max_set_bits_value_index += 1
            
        for i in range(1, max_set_bits_value_index + 1):
            target_set_bits = 1 << i
            count_n_c = count_set_bits_dp(s, target_set_bits)
            count2 += count_n_c
            
        mod = 10**9 + 7
        if k == 1:
            result = (1 + count1) % mod
        else:
            result = (1 + count1 + (k - 1) * count2) % mod
            
        return result