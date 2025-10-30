class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # The minimum possible value of nums[n - 1] will be the result of (n-1) left shifts of x
        # This is because each element in the array must be greater than the previous one
        # and the bitwise AND of all elements must be x.
        return x << (n - 1)