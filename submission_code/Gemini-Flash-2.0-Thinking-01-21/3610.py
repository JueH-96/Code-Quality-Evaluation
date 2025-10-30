from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            counts = Counter(subarray)
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
            top_x_elements = []
            for j in range(min(x, len(sorted_counts))):
                top_x_elements.append(sorted_counts[j][0])
            top_x_set = set(top_x_elements)
            current_x_sum = 0
            for num in subarray:
                if num in top_x_set:
                    current_x_sum += num
            answer.append(current_x_sum)
        return answer