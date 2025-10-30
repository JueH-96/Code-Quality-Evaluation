from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            sub_array = nums[i:i+k]
            counts = Counter(sub_array)
            sorted_counts = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
            
            top_x_elements = [item[0] for item in sorted_counts[:x]]
            
            current_sum = 0
            for num in sub_array:
                if num in top_x_elements:
                    current_sum += num
            result.append(current_sum)
        return result