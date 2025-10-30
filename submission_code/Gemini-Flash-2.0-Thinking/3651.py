class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n

        for i in range(n):
            if nums[i] > 0:
                target_index = (i + nums[i]) % n
                result[i] = nums[target_index]
            elif nums[i] < 0:
                target_index = (i - abs(nums[i]) + n) % n
                result[i] = nums[target_index]
            else:
                result[i] = nums[i]

        return result