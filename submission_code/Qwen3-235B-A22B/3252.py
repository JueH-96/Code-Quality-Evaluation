from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # Extract left and right parts
                left = nums[:i]
                right = nums[j+1:]
                
                # Check if left is strictly increasing
                left_ok = True
                for k in range(len(left) - 1):
                    if left[k] >= left[k+1]:
                        left_ok = False
                        break
                if not left_ok:
                    continue
                
                # Check if right is strictly increasing
                right_ok = True
                for k in range(len(right) - 1):
                    if right[k] >= right[k+1]:
                        right_ok = False
                        break
                if not right_ok:
                    continue
                
                # Check cross condition
                cross_ok = True
                if left and right:
                    if left[-1] >= right[0]:
                        cross_ok = False
                
                if cross_ok:
                    count += 1
        
        return count