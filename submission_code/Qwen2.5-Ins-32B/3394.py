class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Find the smallest number y such that y & (y-1) & ... & (y-(n-1)) == x
        # Start with x and keep adding the smallest bit that is not set in x
        # until we have added n-1 bits.
        result = x
        bit = 1
        while n > 1:
            if result & bit == 0:
                result |= bit
                n -= 1
            bit <<= 1
        return result