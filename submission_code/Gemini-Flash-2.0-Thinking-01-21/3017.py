class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n, k_val):
            s_n = str(n)
            even_digits = 0
            odd_digits = 0
            for digit_char in s_n:
                digit = int(digit_char)
                if digit % 2 == 0:
                    even_digits += 1
                else:
                    odd_digits += 1
            return even_digits == odd_digits and n % k_val == 0
            
        def count_beautiful_in_range(limit, k_val):
            if limit < 1:
                return 0
            limit_str = str(limit)
            n_digits = len(limit_str)
            memo = {}
            
            def solve(index, even_count, remainder, is_tight):
                if index == n_digits:
                    return 1 if even_count * 2 == n_digits and remainder == 0 else 0
                if (index, even_count, remainder, is_tight) in memo:
                    return memo[(index, even_count, remainder, is_tight)]
                
                count = 0
                upper_bound = int(limit_str[index]) if is_tight else 9
                start_digit = 0 if index > 0 else 1
                for digit in range(start_digit, upper_bound + 1):
                    next_tight = is_tight and (digit == upper_bound)
                    next_even_count = even_count + (1 if digit % 2 == 0 else 0)
                    next_remainder = (remainder * 10 + digit) % k_val
                    count += solve(index + 1, next_even_count, next_remainder, next_tight)
                    
                memo[(index, even_count, remainder, is_tight)] = count
                return count
                
            total_count = 0
            for length in range(2, n_digits):
                if length % 2 == 0:
                    memo_len = {}
                    def solve_len(index, even_count, remainder):
                        if index == length:
                            return 1 if even_count * 2 == length and remainder == 0 else 0
                        if (index, even_count, remainder) in memo_len:
                            return memo_len[(index, even_count, remainder)]
                        
                        count = 0
                        start_digit_len = 0 if index > 0 else 1
                        for digit in range(start_digit_len, 10):
                            next_even_count = even_count + (1 if digit % 2 == 0 else 0)
                            next_remainder = (remainder * 10 + digit) % k_val
                            count += solve_len(index + 1, next_even_count, next_remainder)
                            
                        memo_len[(index, even_count, remainder)] = count
                        return count
                    total_count += solve_len(0, 0, 0)

            if n_digits >= 2 and n_digits % 2 == 0:
                total_count += solve(0, 0, 0, True)
                
            return total_count

        return count_beautiful_in_range(high, k) - count_beautiful_in_range(low - 1, k)