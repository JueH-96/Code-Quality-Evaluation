class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        
        # Helper function to count set bits
        def count_set_bits(x):
            count = 0
            while x > 0:
                count += x & 1
                x >>= 1
            return count
        
        # Check if a number is k-reducible
        def is_k_reducible(x):
            steps = 0
            while x > 1 and steps < k:
                x = count_set_bits(x)
                steps += 1
            return x == 1
        
        # Count k-reducible numbers less than n
        count = 0
        for i in range(1, n):
            if is_k_reducible(i):
                count = (count + 1) % MOD
        
        return count