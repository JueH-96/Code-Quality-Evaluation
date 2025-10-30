class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        
        def count_bits(x):
            return bin(x).count('1')
        
        def is_k_reducible(x, k):
            if x == 1:
                return True
            if k == 0:
                return False
                
            next_num = count_bits(x)
            return is_k_reducible(next_num, k-1)
        
        # Handle edge case when n = 1
        if n == 1:
            return 0
            
        count = 0
        # Check all numbers from 1 to n-1
        for i in range(1, n):
            if is_k_reducible(i, k):
                count = (count + 1) % MOD
                
        return count