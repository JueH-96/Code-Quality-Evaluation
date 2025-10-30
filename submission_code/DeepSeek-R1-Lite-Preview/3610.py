from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            freq = Counter(subarray)
            sorted_elements = sorted(freq.keys(), key=lambda elem: (-freq[elem], -elem))
            top_x = sorted_elements[:x]
            sum_x = sum(num for num in subarray if num in top_x)
            answer.append(sum_x)
        return answer