from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        
        for i in range(n - k + 1):
            window = nums[i:i+k]
            freq = Counter(window)
            
            # sort distinct elements by (frequency desc, value desc)
            sorted_keys = sorted(freq.keys(), key=lambda v: (freq[v], v), reverse=True)
            top_x = sorted_keys[:x]
            
            # sum all occurrences of the top-x elements
            s = 0
            for v in top_x:
                s += v * freq[v]
            answer.append(s)
        
        return answer