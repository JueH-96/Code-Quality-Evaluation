class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # The minimum value for nums[0] is x itself
        # We need to find the smallest nums[n-1] such that all elements AND to x
        # Since nums[i] < nums[i+1], we can start from x and keep adding the smallest bit in x not yet included
        
        # Start with the first element as x
        current = x
        # We need n-1 more elements, each greater than the last
        for _ in range(n - 1):
            # Find the smallest bit not included in current that can be added without changing the AND result to x
            bit = 1
            while True:
                # Check if this bit can be added without changing the AND result
                if (current | bit) & x == x:
                    current |= bit
                    break
                bit <<= 1
        
        return current