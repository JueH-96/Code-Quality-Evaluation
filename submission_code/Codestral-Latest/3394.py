class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # The minimum value of nums[n - 1] is x + (n - 1)
        return x + (n - 1)