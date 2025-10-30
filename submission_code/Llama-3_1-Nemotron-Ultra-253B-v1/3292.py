from typing import List
from bisect import bisect_left
from collections import defaultdict

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        sum_d = sum(nums)
        
        # Preprocess changeIndices into a dictionary
        change_dict = defaultdict(list)
        for s in range(m):
            i = changeIndices[s]
            change_dict[i].append(s + 1)  # Convert to 1-based index
        
        # Check if all indices from 1 to n are present
        for i in range(1, n + 1):
            if i not in change_dict:
                return -1
        
        # Check if total operations exceed m
        if sum_d + n > m:
            return -1
        
        # Sort the lists in change_dict
        for i in change_dict:
            change_dict[i].sort()
        
        # Precompute the earliest possible s_i for each i
        earliest_s = []
        for i in range(1, n + 1):
            d_i = nums[i - 1]
            lst = change_dict[i]
            idx = bisect_left(lst, d_i + 1)
            if idx >= len(lst):
                return -1
            earliest_s.append(lst[idx])
        
        low = max(earliest_s)
        high = m
        ans = -1
        
        # Binary search between low and high
        while low <= high:
            mid = (low + high) // 2
            if sum_d + n > mid:
                low = mid + 1
                continue
            
            valid = True
            sum_s = 0
            for i in range(1, n + 1):
                d_i = nums[i - 1]
                lst = change_dict[i]
                idx = bisect_left(lst, d_i + 1)
                if idx >= len(lst) or lst[idx] > mid:
                    valid = False
                    break
                sum_s += lst[idx]
            
            if not valid:
                low = mid + 1
                continue
            
            if sum_d <= sum_s - n:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans if ans != -1 else -1