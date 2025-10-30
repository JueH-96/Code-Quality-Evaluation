from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        result = []
        n = len(nums)
        for i in range(n - k + 1):
            current_window = nums[i:i + k]
            freq = Counter(current_window)
            # Sort by descending frequency, then descending value
            sorted_elements = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            # Take up to x elements
            selected = sorted_elements[:x]
            # Create a set of selected elements for quick lookup
            selected_elements = {element for element, count in selected}
            # Sum elements in the window that are in the selected set
            total = sum(num for num in current_window if num in selected_elements)
            result.append(total)
        return result