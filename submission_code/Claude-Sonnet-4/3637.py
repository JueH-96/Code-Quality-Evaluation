class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        
        MOD = 10**9 + 7
        n = len(num)
        
        # Count frequency of each digit
        digit_count = [0] * 10
        total_sum = 0
        for char in num:
            digit = int(char)
            digit_count[digit] += 1
            total_sum += digit
        
        # If total sum is odd, no balanced permutation exists
        if total_sum % 2 == 1:
            return 0
        
        target_sum = total_sum // 2
        even_positions = (n + 1) // 2  # number of even indices (0, 2, 4, ...)
        odd_positions = n // 2         # number of odd indices (1, 3, 5, ...)
        
        # Precompute factorials and inverse factorials for combinations
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        def mod_inverse(a, mod):
            return pow(a, mod - 2, mod)
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = mod_inverse(fact[n], MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD
        
        # DP: dp[digit][even_used][odd_used][current_sum] = number of ways
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(digit, even_used, odd_used, current_sum):
            if digit == 10:
                if even_used == even_positions and odd_used == odd_positions and current_sum == target_sum:
                    return 1
                return 0
            
            if digit_count[digit] == 0:
                return dp(digit + 1, even_used, odd_used, current_sum)
            
            result = 0
            # Try all possible ways to distribute current digit
            for even_take in range(min(digit_count[digit], even_positions - even_used) + 1):
                for odd_take in range(min(digit_count[digit] - even_take, odd_positions - odd_used) + 1):
                    if even_take + odd_take == digit_count[digit]:
                        # Calculate number of ways to arrange these digits
                        ways = comb(even_positions - even_used, even_take) * comb(odd_positions - odd_used, odd_take) % MOD
                        new_sum = current_sum + digit * even_take
                        result = (result + ways * dp(digit + 1, even_used + even_take, odd_used + odd_take, new_sum)) % MOD
            
            return result
        
        return dp(0, 0, 0, 0)