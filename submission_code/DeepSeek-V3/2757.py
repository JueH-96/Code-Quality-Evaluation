class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_dp(num, min_sum, max_sum):
            n = len(num)
            dp = {}
            
            def dfs(index, tight, sum_digits):
                if index == n:
                    return 1 if min_sum <= sum_digits <= max_sum else 0
                if (index, tight, sum_digits) in dp:
                    return dp[(index, tight, sum_digits)]
                
                limit = int(num[index]) if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_sum = sum_digits + d
                    if new_sum > max_sum:
                        continue
                    total += dfs(index + 1, new_tight, new_sum)
                    total %= MOD
                
                dp[(index, tight, sum_digits)] = total
                return total
            
            return dfs(0, True, 0)
        
        def subtract_one(num):
            num_list = list(num)
            i = len(num_list) - 1
            while i >= 0 and num_list[i] == '0':
                num_list[i] = '9'
                i -= 1
            if i >= 0:
                num_list[i] = str(int(num_list[i]) - 1)
            return ''.join(num_list).lstrip('0') or '0'
        
        count_num2 = digit_dp(num2, min_sum, max_sum)
        count_num1_minus_1 = digit_dp(subtract_one(num1), min_sum, max_sum)
        
        return (count_num2 - count_num1_minus_1) % MOD