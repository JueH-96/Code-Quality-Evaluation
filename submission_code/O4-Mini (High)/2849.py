from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        from bisect import bisect_left
        
        n = len(nums)
        if n <= 1:
            return 0
        
        # Since nums[i] <= n, we can size our presence array to max(nums)+2
        maxv = max(nums)
        ans = 0
        
        # For each starting index i, build subarrays [i..j]
        for i in range(n):
            # presence flag for values seen in this window
            pres = [0] * (maxv + 2)
            # sorted list of distinct values in the current window
            sarr = []
            # current imbalance for window starting at i
            imbal = 0
            
            # Expand the window to the right
            for j in range(i, n):
                v = nums[j]
                # Only do work if this value is new to the window
                if pres[v] == 0:
                    idx = bisect_left(sarr, v)
                    
                    # Find neighbors in the sorted list
                    left = sarr[idx-1] if idx > 0 else None
                    right = sarr[idx] if idx < len(sarr) else None
                    
                    # If left and right both exist, we may remove an existing gap
                    if left is not None and right is not None and right - left > 1:
                        imbal -= 1
                    # Add new gaps introduced by inserting v
                    if left is not None and v - left > 1:
                        imbal += 1
                    if right is not None and right - v > 1:
                        imbal += 1
                    
                    # Mark v present and insert into sorted list
                    pres[v] = 1
                    sarr.insert(idx, v)
                
                # Accumulate the imbalance of the current window [i..j]
                ans += imbal
        
        return ans