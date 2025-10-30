from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        """
        Calculates the minimum average by repeatedly pairing the smallest and largest elements.
        """
        
        # First, sort the array to easily access the smallest and largest elements.
        nums.sort()
        
        n = len(nums)
        
        # A generator expression is used to compute the averages on-the-fly.
        # For each i from 0 to n/2 - 1, we pair the i-th smallest element (nums[i])
        # with the i-th largest element (nums[n-1-i]).
        # The min() function then finds the minimum value from the generated averages.
        # Using 2.0 ensures the result of the division is a float.
        return min((nums[i] + nums[n - 1 - i]) / 2.0 for i in range(n // 2))