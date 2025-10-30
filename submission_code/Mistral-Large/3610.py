from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def x_sum(subarray):
            count = Counter(subarray)
            sorted_items = sorted(count.items(), key=lambda item: (item[1], item[0]), reverse=True)
            top_x = sorted_items[:x]
            return sum(freq * val for val, freq in top_x)

        n = len(nums)
        result = []

        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            result.append(x_sum(subarray))

        return result