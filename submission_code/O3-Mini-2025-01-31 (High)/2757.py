class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        from functools import lru_cache
        
        # Helper: count numbers in the range [0, num] whose digit sum is between min_sum and max_sum.
        def count_up_to(num: str) -> int:
            digits = list(map(int, num))
            n = len(digits)
            
            @lru_cache(maxsize=None)
            def dp(pos: int, current_sum: int, tight: bool) -> int:
                # If the running sum exceeds max_sum, no extension can bring it back into range.
                if current_sum > max_sum:
                    return 0
                # If we've processed all digits, check if we fall in the allowed sum range.
                if pos == n:
                    return 1 if min_sum <= current_sum <= max_sum else 0
                
                # Decide the maximum digit we can place at this position.
                limit = digits[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    total = (total + dp(pos + 1, current_sum + d, tight and (d == limit))) % MOD
                return total
            
            return dp(0, 0, True)
        
        # Helper: subtract one from a numeric string
        def decrement(num: str) -> str:
            num_list = list(num)
            i = len(num_list) - 1
            while i >= 0:
                if num_list[i] != '0':
                    num_list[i] = str(int(num_list[i]) - 1)
                    break
                else:
                    num_list[i] = '9'
                    i -= 1
            # Remove any leading zeros if present (but if empty, return "0")
            res = ''.join(num_list).lstrip('0')
            return res if res != "" else "0"
        
        # Count good numbers in [0, num2] and subtract count of good numbers in [0, num1-1].
        count_num2 = count_up_to(num2)
        count_before_num1 = count_up_to(decrement(num1))
        return (count_num2 - count_before_num1) % MOD