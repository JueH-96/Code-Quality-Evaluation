class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i, num in enumerate(nums):
            if num == 0:
                result[i] = num
            elif num > 0:
                new_index = (i + num) % n
                result[i] = nums[new_index]
            else:
                new_index = (i + num) % n  # since num is negative, adding ensures left stepping
                result[i] = nums[new_index]
        return result