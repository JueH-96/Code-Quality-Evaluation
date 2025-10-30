class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        
        @cache
        def is_k_reducible(x, remaining_k):
            if x == 1:
                return True
            if remaining_k == 0:
                return False
            return is_k_reducible(bin(x).count('1'), remaining_k - 1)
        
        @cache
        def count_reducible(index, tight, current, remaining_k):
            if index == len(s):
                return 1 if is_k_reducible(current, remaining_k) else 0
            
            limit = int(s[index]) if tight else 1
            result = 0
            
            for digit in range(limit + 1):
                new_tight = tight and digit == limit
                new_current = (current << 1) | digit
                result += count_reducible(index + 1, new_tight, new_current, remaining_k)
                result %= MOD
            
            return result
        
        return (count_reducible(0, True, 0, k) - 1 + MOD) % MOD  # Subtract 1 to exclude n itself