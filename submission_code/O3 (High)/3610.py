from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        res = []
        
        for left in range(n - k + 1):                 # every starting index of the window
            window = nums[left:left + k]             # current window of size k
            freq = Counter(window)                   # frequency of each element
            
            # sort by: (1) higher frequency, (2) higher value
            ordered = sorted(freq.items(), key=lambda t: (-t[1], -t[0]))
            
            # pick the top-x (or all if fewer than x)
            top_x = ordered[:x]
            
            # x-sum = Σ value * frequency for the chosen elements
            window_sum = sum(val * cnt for val, cnt in top_x)
            res.append(window_sum)
        
        return res