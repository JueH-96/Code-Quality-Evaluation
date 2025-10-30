from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        # incPrefix[i] : nums[0..i] is strictly increasing
        incPrefix = [False] * n
        incPrefix[0] = True
        for i in range(1, n):
            incPrefix[i] = incPrefix[i-1] and nums[i-1] < nums[i]

        # incSuffix[i] : nums[i..n-1] is strictly increasing
        incSuffix = [False] * n
        incSuffix[-1] = True
        for i in range(n-2, -1, -1):
            incSuffix[i] = incSuffix[i+1] and nums[i] < nums[i+1]

        ans = 0
        for i in range(n):          # start index of subarray to remove
            for j in range(i, n):   # end index of subarray to remove
                # 1. left part (prefix) must be strictly increasing
                if i != 0 and not incPrefix[i-1]:
                    continue
                # 2. right part (suffix) must be strictly increasing
                if j != n-1 and not incSuffix[j+1]:
                    continue
                # 3. boundary condition between left and right parts
                if i != 0 and j != n-1 and nums[i-1] >= nums[j+1]:
                    continue
                ans += 1
        return ans