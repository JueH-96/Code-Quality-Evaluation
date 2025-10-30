from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            count = Counter(sub)
            unique_elements = list(count.keys())
            # Sort by descending frequency, then descending element value
            sorted_elements = sorted(unique_elements, key=lambda e: (-count[e], -e))
            selected = sorted_elements[:x]
            selected_set = set(selected)
            total = sum(num for num in sub if num in selected_set)
            result.append(total)
        return result