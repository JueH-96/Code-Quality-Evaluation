class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Try different values of k (number of operations)
        for k in range(1, 65):  # We need at most 64 operations (one for each bit position)
            target = num1 - k * num2
            
            # Check if target is positive
            if target <= 0:
                continue
            
            # Check if we can represent target as sum of k powers of 2
            # Minimum: k distinct smallest powers = 2^0 + 2^1 + ... + 2^(k-1) = 2^k - 1
            # Maximum: k times the largest power = k * 2^60
            
            # Also, we need at least popcount(target) powers of 2 to represent target
            bit_count = bin(target).count('1')
            
            # We need: bit_count <= k <= target
            # bit_count <= k: we need at least bit_count powers to represent target
            # k <= target: each power is at least 2^0 = 1, so sum of k powers is at least k
            
            if bit_count <= k <= target:
                return k
        
        return -1