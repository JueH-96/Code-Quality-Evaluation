from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        answer = []
        n = len(nums)
        for i in range(n - k + 1):
            window = nums[i:i+k]
            counts = Counter(window)
            # Sort elements by frequency descending, then value descending
            sorted_elements = sorted(counts.keys(), key=lambda num: (-counts[num], -num))
            # Take top x elements
            top_x = sorted_elements[:x]
            # Calculate the sum of their occurrences
            total = sum(counts[num] * num for num in top_x)
            answer.append(total)
        return answer