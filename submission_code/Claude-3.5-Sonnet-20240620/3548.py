class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        def is_palindrome(num):
            return str(num) == str(num)[::-1]
        
        @lru_cache(None)
        def count_palindromes(length, sum_mod_k, leading_zero):
            if length == 0:
                return 1 if sum_mod_k == 0 else 0
            
            total = 0
            start = 0 if leading_zero else 1
            for digit in range(start, 10):
                total += count_palindromes(length - 1, (sum_mod_k - digit) % k, False)
            
            return total % MOD
        
        total = 0
        for length in range(1, n + 1):
            if length % 2 == 0:
                half_length = length // 2
                total += count_palindromes(half_length, 0, False)
            else:
                half_length = length // 2
                for middle_digit in range(10):
                    total += count_palindromes(half_length, (k - middle_digit % k) % k, False)
            total %= MOD
        
        return total