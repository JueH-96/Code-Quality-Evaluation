from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []

        for i in range(n - k + 1):
            subarray = nums[i : i + k]
            counts = Counter(subarray)

            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))

            top_x_elements = [item[0] for item in sorted_counts[:x]]

            current_x_sum = 0
            for element in subarray:
                if element in top_x_elements:
                    current_x_sum += element

            result.append(current_x_sum)

        return result