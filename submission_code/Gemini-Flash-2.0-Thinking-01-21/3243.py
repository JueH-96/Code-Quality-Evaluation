class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def get_powerful_count(n, limit_val, suffix_s):
            if n < 1:
                return 0
            n_str = str(n)
            n_len = len(n_str)
            s_len = len(suffix_s)
            total_count = 0
            
            for length in range(s_len, n_len):
                if length == s_len:
                    is_powerful_suffix = True
                    for digit_char in suffix_s:
                        if int(digit_char) > limit_val:
                            is_powerful_suffix = False
                            break
                    if is_powerful_suffix:
                        total_count += 1
                else:
                    count_for_length = 1
                    if length - s_len > 0:
                        count_for_length = limit_val
                        if length - s_len - 1 > 0:
                            count_for_length *= (limit_val + 1) ** (length - s_len - 1)
                    is_powerful_suffix = True
                    for digit_char in suffix_s:
                        if int(digit_char) > limit_val:
                            is_powerful_suffix = False
                            break
                    if is_powerful_suffix:
                        total_count += count_for_length
                        
            memo = {}
            
            def solve_dp(index, is_tight):
                if index == n_len:
                    return 1
                if (index, is_tight) in memo:
                    return memo[(index, is_tight)]
                
                count = 0
                upper_bound = int(n_str[index]) if is_tight else limit_val
                
                start_digit = 0
                if index == 0 and n_len > 1:
                    start_digit = 1
                    
                for digit in range(start_digit, upper_bound + 1):
                    if digit > limit_val:
                        continue
                    next_tight = is_tight and (digit == upper_bound)
                    
                    current_index_from_suffix_start = index - (n_len - s_len)
                    if current_index_from_suffix_start >= 0:
                        required_digit = int(suffix_s[current_index_from_suffix_start])
                        if digit != required_digit:
                            continue
                            
                    count += solve_dp(index + 1, next_tight)
                    
                memo[(index, is_tight)] = count
                return count
                
            count_for_n_len = solve_dp(0, True)
            total_count += count_for_n_len
            
            return total_count

        start_val = max(1, start)
        count_finish = get_powerful_count(finish, limit, s)
        count_start_minus_one = get_powerful_count(start_val - 1, limit, s)
        
        return count_finish - count_start_minus_one