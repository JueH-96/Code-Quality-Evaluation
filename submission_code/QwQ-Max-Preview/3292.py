from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        left = 1
        right = m
        ans = -1
        
        while left <= right:
            mid = (left + right) // 2
            possible = True
            last_occurrence = [-1] * n
            
            # Populate last_occurrence with the last occurrence of each index up to mid steps
            for k in range(mid):
                idx = changeIndices[k] - 1  # Convert 1-based to 0-based
                last_occurrence[idx] = k + 1  # steps are 1-based
            
            # Check if all indices have at least one occurrence
            for lo in last_occurrence:
                if lo == -1:
                    possible = False
                    break
            if not possible:
                left = mid + 1
                continue
            
            # Check sum(nums) + n <= mid
            sum_nums = sum(nums) + n
            if sum_nums > mid:
                possible = False
            else:
                # Check each index's last occurrence is >= nums[i] + 1
                for i in range(n):
                    required = nums[i] + 1
                    if last_occurrence[i] < required:
                        possible = False
                        break
            
            if possible:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans if ans != -1 else -1