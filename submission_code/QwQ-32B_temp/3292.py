import bisect
from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if all indices from 1 to n are present in changeIndices
        present = set(changeIndices)
        for i in range(1, n + 1):
            if i not in present:
                return -1
        
        # Collect occurrences for each index
        occurrences = [[] for _ in range(n + 1)]  # 1-based indexing
        for s in range(1, m + 1):
            idx = changeIndices[s - 1]
            occurrences[idx].append(s)
        
        s_list = []
        for i in range(1, n + 1):
            required = nums[i - 1] + 1
            occ_list = occurrences[i]
            
            # Find the first occurrence >= required using bisect
            idx = bisect.bisect_left(occ_list, required)
            if idx < len(occ_list):
                s_i = occ_list[idx]
            else:
                return -1
            s_list.append(s_i)
        
        s_candidate = max(s_list)
        total_needed = sum(nums) + n
        required_s = max(s_candidate, total_needed)
        
        if required_s <= m:
            return required_s
        else:
            return -1