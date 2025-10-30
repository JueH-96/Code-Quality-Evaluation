class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        # Check each 1-indexed position
        for i in range(1, n + 1):
            # If i divides n, then nums[i] is special
            if n % i == 0:
                # Convert to 0-indexed to access the array
                total += nums[i - 1] ** 2
        
        return total