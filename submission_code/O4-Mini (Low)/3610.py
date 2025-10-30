from typing import List
import collections

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            window = nums[i:i+k]
            counts = collections.Counter(window)
            
            # If fewer distinct elements than x, just sum the whole window
            if len(counts) <= x:
                result.append(sum(window))
                continue
            
            # Build list of (freq, value) and sort by freq desc, then value desc
            freq_vals = [ (cnt, val) for val, cnt in counts.items() ]
            freq_vals.sort(key=lambda cv: (cv[0], cv[1]), reverse=True)
            
            # Sum up the top x frequent elements: freq * value
            x_sum = 0
            for j in range(x):
                cnt, val = freq_vals[j]
                x_sum += cnt * val
            
            result.append(x_sum)
        
        return result