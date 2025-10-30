class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n_bits = len(s)
        
        # Precompute operations[i] = number of operations to reduce the number i to 1
        operations = [0] * (n_bits + 1)
        operations[1] = 0
        for i in range(2, n_bits + 1):
            operations[i] = 1 + operations[bin(i).count('1')]
        
        # Determine which bit counts lead to k-reducible numbers
        # A number with b set bits needs 1 + operations[b] operations
        # (except the number 1 which needs 0 operations, but since k >= 1, it's always k-reducible)
        k_reducible_bits = set()
        for b in range(1, n_bits + 1):
            if 1 + operations[b] <= k:
                k_reducible_bits.add(b)
        
        # Use digit DP to count numbers less than n with k-reducible bit counts
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(pos, bit_count, is_limit):
            if pos == n_bits:
                return 1 if bit_count in k_reducible_bits else 0
            
            max_digit = int(s[pos]) if is_limit else 1
            result = 0
            
            for digit in range(max_digit + 1):
                result += dp(pos + 1, bit_count + digit, is_limit and (digit == max_digit))
                result %= MOD
            
            return result
        
        return dp(0, 0, True)