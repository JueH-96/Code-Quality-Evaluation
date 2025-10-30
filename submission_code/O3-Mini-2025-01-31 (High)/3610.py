from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        
        # Process each contiguous subarray (window) of size k.
        for i in range(n - k + 1):
            # Get the current window
            window = nums[i:i+k]
            # Count frequencies of each element in the window.
            freq = Counter(window)
            
            # Sort items by frequency descending, and if frequencies are equal, sort by value descending.
            # Each item is (element, count) so we sort by (count, element) in reverse order.
            sorted_items = sorted(freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
            
            # If there are less than x distinct numbers, we take all, otherwise take the top x.
            top_x = sorted_items[:x]
            
            # Compute the x-sum: for each selected element, multiply it by its frequency and add.
            x_sum = sum(val * count for val, count in top_x)
            result.append(x_sum)
            
        return result