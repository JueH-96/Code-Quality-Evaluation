class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We'll try different values of k
        for k in range(1, 61):  # Max 60 bits needed for constraints
            target = num1 - k * num2
            
            # Target must be positive
            if target <= 0:
                break
                
            # Check if target can be expressed as a sum of k powers of 2
            bit_count = bin(target).count('1')
            
            # Two conditions: need at most k bits set to 1, and target ≥ k
            if bit_count <= k and target >= k:
                return k
                
        return -1