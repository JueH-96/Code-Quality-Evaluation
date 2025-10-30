from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            window = nums[i:i+k]
            freq = Counter(window)
            # Sort elements by frequency (descending), then by value (descending)
            sorted_elements = sorted(freq.keys(), key=lambda e: (-freq[e], -e))
            # Select top x elements
            selected_elements = sorted_elements[:x]
            selected_set = set(selected_elements)
            # Calculate the sum of elements in the window that are among the selected elements
            current_sum = sum(num for num in window if num in selected_set)
            result.append(current_sum)
        return result