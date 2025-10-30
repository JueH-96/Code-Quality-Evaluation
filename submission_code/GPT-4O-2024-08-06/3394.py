class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Start with the smallest possible number which is x itself
        current = x
        # We need n numbers, so we need to find the (n-1)th number greater than x
        for i in range(n - 1):
            # Increment current until we find a number whose AND with x is still x
            while (current & x) != x:
                current += 1
            # Move to the next number
            current += 1
        # The last valid number we found is the minimum possible value for nums[n-1]
        return current - 1