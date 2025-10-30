from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)

        # The outlier must be one of the two largest values
        candidate1 = nums[-1]
        candidate2 = nums[-2]

        # Check if candidate1 can be the outlier
        if candidate1 != nums[-2] + nums[-3]:
            special_nums = nums[:-1]
            if candidate1 == sum(special_nums):
                return candidate2
            else:
                return candidate1

        # Check if candidate2 can be the outlier
        if candidate2 != nums[-1] + nums[-3]:
            special_nums = nums[:-2] + nums[-1:]
            if candidate2 == sum(special_nums):
                return candidate1
            else:
                return candidate2

        # If neither candidate1 nor candidate2 is the outlier, return the larger one
        return max(candidate1, candidate2)