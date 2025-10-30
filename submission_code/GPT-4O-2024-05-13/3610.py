from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def x_sum(subarray: List[int], x: int) -> int:
            freq = Counter(subarray)
            # Sort by frequency first (descending), then by value (descending)
            sorted_elements = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            # Take the top x elements
            top_x_elements = sorted_elements[:x]
            # Calculate the sum of the subarray considering only the top x elements
            top_x_set = {item[0] for item in top_x_elements}
            return sum(num for num in subarray if num in top_x_set)
        
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            result.append(x_sum(subarray, x))
        
        return result