class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Cache for is_at_most_j_reducible
        is_reducible = {}
        
        def is_at_most_j_reducible(num, j):
            if num == 1:
                return True
            if j == 0:
                return False
            if (num, j) in is_reducible:
                return is_reducible[(num, j)]
            
            next_num = bin(num).count('1')
            result = is_at_most_j_reducible(next_num, j-1)
            is_reducible[(num, j)] = result
            return result
        
        # Digit-by-digit DP to count the number of integers less than n
        memo = {}
        
        def dp(index, tight, count_bits, non_zero):
            if index == n:
                return 1 if non_zero and is_at_most_j_reducible(count_bits, k-1) else 0
            
            if (index, tight, count_bits, non_zero) in memo:
                return memo[(index, tight, count_bits, non_zero)]
            
            limit = int(s[index]) if tight else 1
            result = 0
            
            for digit in range(limit + 1):
                new_tight = tight and digit == limit
                new_count_bits = count_bits + digit
                new_non_zero = non_zero or digit > 0
                
                result = (result + dp(index + 1, new_tight, new_count_bits, new_non_zero)) % MOD
            
            memo[(index, tight, count_bits, non_zero)] = result
            return result
        
        return dp(0, True, 0, False)