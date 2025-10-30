class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Precompute minimum operations needed to reduce a number with 'bits' set bits to 1
        min_ops = [float('inf')] * (n + 1)
        min_ops[1] = 0  # 1 is already 1
        
        for bits in range(2, n + 1):
            next_bits = bin(bits).count('1')
            if next_bits < bits:
                min_ops[bits] = min_ops[next_bits] + 1
        
        # Count numbers less than the given binary string with specific bit counts
        from functools import lru_cache
        
        @lru_cache(None)
        def count_with_bits(pos, bits_count, is_limit):
            if pos == n:
                # Check if this bit count is k-reducible
                return 1 if bits_count > 0 and min_ops[bits_count] <= k else 0
            
            limit = int(s[pos]) if is_limit else 1
            result = 0
            
            for digit in range(limit + 1):
                new_limit = is_limit and (digit == limit)
                new_bits = bits_count + digit
                result = (result + count_with_bits(pos + 1, new_bits, new_limit)) % MOD
            
            return result
        
        # Count all valid numbers from 0 to n-1
        total = count_with_bits(0, 0, True)
        
        # Since we want positive integers (excluding 0), we need to check if 0 was counted
        # 0 has 0 set bits, which is not k-reducible for any k
        # So we don't need to subtract anything
        
        return total % MOD