class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_len = len(s)
        
        def count_powerful_up_to(n):
            if n < 0:
                return 0
            n_str = str(n)
            n_len = len(n_str)
            total_count = 0
            memo = {}
            
            def dp(index, is_tight, current_length):
                if index == current_length:
                    return 1
                if (index, is_tight) in memo:
                    return memo[(index, is_tight)]
                
                count = 0
                if index >= current_length - s_len:
                    digit = int(s[index - (current_length - s_len)])
                    upper_bound = digit
                    if is_tight:
                        upper_bound = min(upper_bound, int(n_str[index]))
                    if upper_bound > limit:
                        result = 0
                    else:
                        if digit > limit:
                            result = 0
                        else:
                            next_is_tight = is_tight and (digit == int(n_str[index]) if index < n_len else True)
                            result = dp(index + 1, next_is_tight, current_length)
                    
                else:
                    start_digit = 1 if index == 0 else 0
                    upper_bound_n = 9
                    if is_tight:
                        upper_bound_n = int(n_str[index])
                    upper_bound = min(limit, upper_bound_n)
                    for digit in range(start_digit, upper_bound + 1):
                        next_is_tight = is_tight and (digit == upper_bound_n)
                        count += dp(index + 1, next_is_tight, current_length)
                
                memo[(index, is_tight)] = count
                return count
                
            for length in range(s_len, n_len + 1):
                memo = {}
                count = dp(0, True, length)
                total_count += count
                
            if len(str(n)) < s_len:
                return total_count
                
            integer_s = int(s)
            if integer_s <= n:
                is_powerful_s = True
                for digit_char in s:
                    if int(digit_char) > limit:
                        is_powerful_s = False
                        break
                if is_powerful_s and len(s) <= len(str(n)):
                    total_count += 1 if int(s) <= n else 0
                    
            return total_count - (1 if len(str(n)) >= s_len and int(str(n)) == int(s) and all(int(d) <= limit for d in str(n)) and int(str(n)) <= n else 0) if len(str(n)) == s_len else total_count

        
        start_minus_one = start - 1
        count_finish = count_powerful_up_to(finish)
        count_start_minus_one = count_powerful_up_to(start_minus_one)
        
        return count_finish - count_start_minus_one