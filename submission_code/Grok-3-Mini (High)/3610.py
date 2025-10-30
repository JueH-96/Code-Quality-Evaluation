from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            sub = nums[i:i + k]
            freq = Counter(sub)
            sorted_elements = sorted(freq.keys(), key=lambda el: (-freq[el], -el))
            top_x = sorted_elements[:x]
            sum_x = sum(el * freq[el] for el in top_x)
            answer.append(sum_x)
        return answer