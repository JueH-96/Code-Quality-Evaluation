class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] > 0:
                steps = nums[i]
                new_index = (i + steps) % n
                result[i] = nums[new_index]
            elif nums[i] < 0:
                steps = abs(nums[i])
                new_index = (i - steps) % n
                result[i] = nums[new_index]
            else:
                result[i] = nums[i]
        return result