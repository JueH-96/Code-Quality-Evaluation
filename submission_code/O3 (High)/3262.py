from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the side lengths so every prefix ends with its current largest side
        nums.sort()
        
        prefix_sum = 0           # running perimeter of current prefix
        best = -1                # best (largest) valid perimeter found so far
        
        for i, side in enumerate(nums):
            prefix_sum += side   # extend current prefix
            
            # A polygon needs at least 3 sides, i >= 2 means prefix length >= 3
            # Polygon inequality (for sorted prefix):   prefix_sum - side > side
            # which is equivalent to                   prefix_sum > 2 * side
            if i >= 2 and prefix_sum > 2 * side:
                best = prefix_sum
        
        return best