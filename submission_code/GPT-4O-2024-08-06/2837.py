class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We will try to find the minimum number of operations needed
        # by iterating over possible number of operations `k`.
        # We need to find the smallest `k` such that:
        # num1 - k * num2 = sum of distinct powers of 2 (i.e., a binary number)
        
        for k in range(1, 61):  # We try up to 60 operations
            target = num1 - k * num2
            # Check if target can be represented as a sum of distinct powers of 2
            # This is equivalent to checking if target is a non-negative integer
            # and the number of 1s in its binary representation is <= k
            if target >= 0 and bin(target).count('1') <= k:
                return k
        
        return -1