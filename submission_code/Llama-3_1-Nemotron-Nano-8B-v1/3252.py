from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                # Check left part (0 to i-1)
                left_ok = True
                if i > 0:
                    for k in range(i - 1):
                        if nums[k] >= nums[k + 1]:
                            left_ok = False
                            break
                # Check right part (j+1 to n-1)
                right_ok = True
                if j < n - 1:
                    for k in range(j + 1, n - 1):
                        if nums[k] >= nums[k + 1]:
                            right_ok = False
                            break
                # Check if left and right are ok and the bridge condition
                if left_ok and right_ok:
                    if (i == 0 or j == n - 1) or (nums[i - 1] < nums[j + 1]):
                        count += 1
        return count