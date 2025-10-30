from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        if len(nums) <= 3: return 0
        
        pos = [i for i in range(len(nums)) if nums[i] != -1]
        l, r = pos[0], pos[-1]
        if len(pos) <= 3: return 0
        
        sorted_nums = sorted(nums[l:r+1])
        ans = min(
            sorted_nums[-1] - sorted_nums[3],
            sorted_nums[-4] - sorted_nums[0],
            )
        
        def helper(i, j, nums, pos):
            if isinstance(i, int):
                l = i
                r = j
            else:
                l = pos[i]
                r = pos[j]

            while l+1 < r-1:
                if nums[l] < nums[l+1]:
                    if nums[r-2] > nums[r-1]:
                        return min(nums[r-2] - nums[l+1], nums[r-1]-nums[l], nums[r-1] - nums[l+1])
                    return nums[r-1] - nums[l]
                else:
                    if nums[r-2] > nums[r-1]:
                        return nums[r-2] - nums[l]
                    return min(nums[r-2] - nums[l], nums[r-1]-nums[l-1], nums[r-2]-nums[l-1])
        
        ans = min(ans, helper(0, -1, nums, pos), helper(1, -2, nums, pos))
        return ans