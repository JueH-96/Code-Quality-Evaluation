class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] > 0:
                steps = nums[i]
                index = (i + steps) % n
                result[i] = nums[index]
            elif nums[i] < 0:
                steps = abs(nums[i])
                index = (i - steps) % n
                result[i] = nums[index]
            else:
                result[i] = nums[i]
        return result