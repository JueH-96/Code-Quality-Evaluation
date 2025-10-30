from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        answer = []
        n = len(nums)
        for i in range(n - k + 1):
            window = nums[i:i + k]
            freq_dict = Counter(window)
            unique_elements = list(freq_dict.keys())
            # Sort the elements based on frequency descending and element value descending
            sorted_elements = sorted(unique_elements, key=lambda y: (-freq_dict[y], -y))
            top_x = sorted_elements[:x]
            top_x_set = set(top_x)
            sum_val = 0
            for num in window:
                if num in top_x_set:
                    sum_val += num
            answer.append(sum_val)
        return answer