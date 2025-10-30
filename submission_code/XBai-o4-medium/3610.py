from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            counts = Counter(subarray)
            unique_elements = list(counts.keys())
            # Sort based on frequency descending, then value descending
            sorted_elements = sorted(unique_elements, key=lambda num: (-counts[num], -num))
            selected = sorted_elements[:x]
            selected_set = set(selected)
            current_sum = sum(num for num in subarray if num in selected_set)
            answer.append(current_sum)
        return answer