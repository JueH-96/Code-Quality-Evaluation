from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        result = []
        n = len(nums)
        for i in range(n - k + 1):
            window = nums[i:i + k]
            freq = Counter(window)
            sorted_elements = sorted(freq.keys(), key=lambda num: (-freq[num], -num))
            top_x = sorted_elements[:x]
            total = sum(num for num in window if num in top_x)
            result.append(total)
        return result