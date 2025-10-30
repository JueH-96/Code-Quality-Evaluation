from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def x_sum(arr):
            if len(set(arr)) <= x:
                return sum(arr)
            
            count = Counter(arr)
            top_x = sorted(count.items(), key=lambda item: (-item[1], -item[0]))[:x]
            return sum(num * freq for num, freq in top_x)
        
        n = len(nums)
        answer = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            answer.append(x_sum(subarray))
        
        return answer