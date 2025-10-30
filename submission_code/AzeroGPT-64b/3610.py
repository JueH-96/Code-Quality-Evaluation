from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def x_sum(subarray, k, x):
            freq = Counter(subarray)
            sorted_items = sorted(list(freq.items()), key=lambda item: (-item[1], -item[0]))
            top_x_items = sorted_items[:x]
            filtered_sum = 0
            for key, val in top_x_items:
                filtered_sum += key * val
            return filtered_sum
            
        result = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            result.append(x_sum(subarray, k, x))
        return result