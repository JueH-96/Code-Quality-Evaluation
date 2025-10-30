class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        for i in range(n):
            landing_index = (i + nums[i]) % n
            res[i] = nums[landing_index]
        return res