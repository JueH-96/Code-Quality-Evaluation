from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        nums.sort()
        if nums[-1] == 0:
            return 0
        if nums[0] >= 0:
            product = 1
            for num in nums:
                product *= num
            return product
        if nums[-1] <= 0:
            if len(nums) % 2 == 0:
                product = 1
                for num in nums:
                    product *= num
                return product
            else:
                product = 1
                for num in nums[:-1]:
                    product *= num
                return product
        neg = [x for x in nums if x < 0]
        pos = [x for x in nums if x > 0]
        if len(neg) % 2 != 0:
            neg.pop()
        product = 1
        for num in neg + pos:
            product *= num
        return product

# Example usage:
# sol = Solution()
# print(sol.maxStrength([3,-1,-5,2,5,-9]))  # Output: 1350
# print(sol.maxStrength([-4,-5,-4]))  # Output: 20