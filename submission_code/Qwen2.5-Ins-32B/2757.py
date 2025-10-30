class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_sum(n: str) -> int:
            return sum(int(digit) for digit in n)
        
        def count_good_numbers(n: str, min_sum: int, max_sum: int) -> int:
            memo = {}
            
            def dp(index, is_tight, current_sum):
                if index == len(n):
                    return 1 if min_sum <= current_sum <= max_sum else 0
                if not is_tight and (index, current_sum) in memo:
                    return memo[(index, current_sum)]
                
                limit = int(n[index]) if is_tight else 9
                count = 0
                for digit in range(limit + 1):
                    new_is_tight = is_tight and digit == limit
                    count += dp(index + 1, new_is_tight, current_sum + digit)
                
                if not is_tight:
                    memo[(index, current_sum)] = count % MOD
                return count % MOD
            
            return dp(0, True, 0)
        
        return (count_good_numbers(num2, min_sum, max_sum) - count_good_numbers(str(int(num1) - 1), min_sum, max_sum)) % MOD