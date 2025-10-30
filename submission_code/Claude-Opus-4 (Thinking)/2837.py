class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 64):  # Try different number of operations
            s = num1 - k * num2
            
            # Check if s can be expressed as sum of k powers of 2
            if s < k:
                continue  # Minimum sum with k powers is k
            if s > k * (1 << 60):
                continue  # Maximum sum with k powers is k * 2^60
            
            # Count number of 1-bits in s
            popcount = bin(s).count('1')
            if popcount <= k:
                return k
        
        return -1