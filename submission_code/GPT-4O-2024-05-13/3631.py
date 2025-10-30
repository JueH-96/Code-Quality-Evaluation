class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        
        def count_set_bits(x):
            return bin(x).count('1')
        
        def is_k_reducible(x, k):
            steps = 0
            while x > 1 and steps < k:
                x = count_set_bits(x)
                steps += 1
            return x == 1
        
        n = int(s, 2)
        count = 0
        
        for i in range(1, n):
            if is_k_reducible(i, k):
                count += 1
        
        return count % MOD