class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        
        # Convert binary string to integer
        n = int(s, 2)
        
        # Function to count set bits
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Function to check if a number is k-reducible
        def is_k_reducible(x, k):
            for _ in range(k):
                if x == 1:
                    return True
                x = count_set_bits(x)
            return x == 1
        
        # Count k-reducible numbers less than n
        count = 0
        for i in range(1, n):
            if is_k_reducible(i, k):
                count += 1
                count %= MOD
        
        return count