class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        # Check each 1-indexed position i from 1 to n
        for i in range(1, n + 1):
            if n % i == 0:  # i divides n
                # Convert to 0-indexed for Python array access
                total += nums[i - 1] ** 2
        
        return total