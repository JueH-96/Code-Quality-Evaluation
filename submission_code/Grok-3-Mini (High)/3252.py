from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Compute prefix_ok: prefix_ok[i] is True if subarray from 0 to i is strictly increasing
        prefix_ok = [False] * n
        if n > 0:
            prefix_ok[0] = True
            for i in range(1, n):
                if prefix_ok[i - 1] and nums[i - 1] < nums[i]:
                    prefix_ok[i] = True
                else:
                    prefix_ok[i] = False  # Redundant but kept for clarity
        
        # Compute suffix_ok: suffix_ok[j] is True if subarray from j to n-1 is strictly increasing
        suffix_ok = [False] * n
        if n > 0:
            suffix_ok[n - 1] = True
            for j in range(n - 2, -1, -1):
                if suffix_ok[j + 1] and nums[j] < nums[j + 1]:
                    suffix_ok[j] = True
                else:
                    suffix_ok[j] = False  # Redundant but kept for clarity
        
        # Initialize count of incremovable subarrays
        count = 0
        
        # Iterate over all possible subarrays defined by start index s and end index e
        for s in range(n):
            for e in range(s, n):
                # Check if left part is strictly increasing
                if s == 0:
                    left_ok = True
                else:
                    left_ok = prefix_ok[s - 1]
                
                # Check if right part is strictly increasing
                if e == n - 1:
                    right_ok = True
                else:
                    right_ok = suffix_ok[e + 1]
                
                # If both left and right are okay, check the join condition
                if left_ok and right_ok:
                    if not (s > 0 and e < n - 1) or nums[s - 1] < nums[e + 1]:
                        count += 1
        
        return count