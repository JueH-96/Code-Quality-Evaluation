from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Select the first k elements
        selected = nums[:k]
        
        # Calculate the sum of squares of the selected elements
        sum_of_squares = sum(x * x for x in selected) % MOD
        
        return sum_of_squares