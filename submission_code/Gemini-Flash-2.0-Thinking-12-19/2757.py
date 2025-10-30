class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def count_upto_n(n_str, min_s, max_s):
            n_len = len(n_str)
            memo = {}
            
            def dp(pos, current_sum, tight):
                if pos == n_len:
                    return 1 if min_s <= current_sum <= max_s else 0
                if (pos, current_sum, tight) in memo:
                    return memo[(pos, current_sum, tight)]
                
                count = 0
                limit = int(n_str[pos]) if tight else 9
                for digit in range(limit + 1):
                    next_sum = current_sum + digit
                    if next_sum > max_s:
                        continue
                    next_tight = tight and (digit == limit)
                    count = (count + dp(pos + 1, next_sum, next_tight)) % MOD
                    
                memo[(pos, current_sum, tight)] = count
                return count
                
            return dp(0, 0, True)
            
        n2_str = num2
        n1_minus_1 = int(num1) - 1
        n1_minus_1_str = str(n1_minus_1) if n1_minus_1 >= 0 else "0"
        
        ans = (count_upto_n(n2_str, min_sum, max_sum) - count_upto_n(n1_minus_1_str, min_sum, max_sum)) % MOD
        return ans if ans >= 0 else ans + MOD