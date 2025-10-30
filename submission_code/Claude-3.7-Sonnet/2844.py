class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        
        for i in range(1, n+1):
            if n % i == 0:  # Check if position i divides n evenly
                result += nums[i-1] ** 2  # Convert to 0-indexed for Python
                
        return result