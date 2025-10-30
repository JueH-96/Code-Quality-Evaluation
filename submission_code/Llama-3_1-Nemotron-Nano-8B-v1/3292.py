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
        
        s_list = []
        # For each index i, find the earliest s_i where changeIndices[s] == i and s+1 >= nums[i-1] + 1
        for i in range(1, n + 1):
            target = nums[i - 1] + 1
            found = False
            for s in range(m):
                if changeIndices[s] == i and (s + 1) >= target:
                    s_list.append(s + 1)  # Convert to 1-based index
                    found = True
                    break
            if not found:
                return -1
        
        sum_decrements = sum(nums)
        # Check if total decrements can fit into available time slots (m - n)
        if sum_decrements > (m - n):
            return -1
        
        max_s_i = max(s_list)
        # The answer is the maximum between the latest marking time and the sum of decrements plus the number of indices
        T = max(max_s_i, sum_decrements + n)
        return T