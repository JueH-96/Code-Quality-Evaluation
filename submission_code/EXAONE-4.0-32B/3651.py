class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] == 0:
                result[i] = 0
            else:
                landing_index = (i + nums[i]) % n
                result[i] = nums[landing_index]
        return result