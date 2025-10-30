class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] > 0:
                new_index = (i + nums[i]) % n
                result[i] = nums[new_index]
            elif nums[i] < 0:
                new_index = (i + nums[i]) % n
                result[i] = nums[new_index]
            else:
                result[i] = nums[i]
        return result