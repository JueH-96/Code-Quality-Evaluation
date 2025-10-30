class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Try different numbers of operations
        for k in range(1, 50):  # 50 is sufficient given constraints
            target = num1 - k * num2
            
            # Check if target can be represented as sum of exactly k powers of 2
            if target >= k and bin(target).count('1') <= k:
                return k
        
        return -1