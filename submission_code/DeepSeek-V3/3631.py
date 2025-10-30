class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        
        # Function to count the number of set bits in a number
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Function to check if a number is k-reducible
        def is_k_reducible(x):
            steps = 0
            while x != 1 and steps <= k:
                x = count_set_bits(x)
                steps += 1
            return x == 1 and steps <= k
        
        # Convert binary string to integer
        n = int(s, 2)
        
        # Count the number of k-reducible numbers less than n
        count = 0
        for x in range(1, n):
            if is_k_reducible(x):
                count += 1
                if count >= MOD:
                    count -= MOD
        
        return count % MOD