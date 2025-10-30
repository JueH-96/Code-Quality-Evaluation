class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # For each number of operations k
        # After k operations, num1 = original_num1 - k*num2 - sum(powers of 2)
        # sum(powers of 2) must equal original_num1 - k*num2
        # And we need k bits set to 1 in this sum
        for k in range(1, 61):
            target = num1 - k * num2
            
            # If target is negative or k is greater than target, impossible
            if target <= 0 or k > target:
                return -1
                
            # Count bits in target
            bits = bin(target).count('1')
            
            # We need k bits set to 1, and k must be >= bits
            # But k must also be <= target (since each power of 2 is at least 1)
            if k >= bits and k <= target:
                return k
                
        return -1