class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10 ** 9 + 7
        
        from functools import lru_cache
        
        # Function to compute f(n): count of numbers <= n with digit sum between min_sum and max_sum
        def count_numbers(num):
            n = len(num)
            @lru_cache(None)
            def dfs(pos, sum_, is_num_started, tight):
                if pos == n:
                    if is_num_started and min_sum <= sum_ <= max_sum:
                        return 1
                    else:
                        return 0
                res = 0
                upper_limit = int(num[pos]) if tight else 9
                for digit in range(0, upper_limit + 1):
                    new_is_num_started = is_num_started or (digit != 0)
                    new_sum = sum_ + digit if new_is_num_started else sum_
                    if new_sum > max_sum:
                        continue
                    new_tight = tight and (digit == upper_limit)
                    res += dfs(pos + 1, new_sum, new_is_num_started, new_tight)
                return res % MOD
            return dfs(0, 0, False, True)

        # Function to decrement num_str by 1
        def decrement(num_str):
            num_list = list(num_str)
            idx = len(num_list) - 1
            while idx >= 0 and num_list[idx] == '0':
                num_list[idx] = '9'
                idx -= 1
            if idx == -1:
                return ''  # num_str was '0' or all zeros
            num_list[idx] = str(int(num_list[idx]) -1)
            # Remove leading zeros
            result = ''.join(num_list).lstrip('0')
            return result if result != '' else '0'

        num1_minus_one = decrement(num1)
        count_num2 = count_numbers(num2)
        count_num1_minus_one = count_numbers(num1_minus_one) if num1_minus_one != '' and num1_minus_one != '0' else 0
        result = (count_num2 - count_num1_minus_one) % MOD
        return result