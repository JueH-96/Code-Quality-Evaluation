class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        sum_squares = 0
        
        # Check each 1-indexed position from 1 to n
        for i in range(1, n + 1):
            # If i divides n, then position i is special
            if n % i == 0:
                # Add the square of element at position i (convert to 0-indexed)
                sum_squares += nums[i - 1] ** 2
        
        return sum_squares