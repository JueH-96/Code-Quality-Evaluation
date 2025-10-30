class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n):
            if nums[i] > 0:
                new_i = (i + nums[i]) % n
            elif nums[i] < 0:
                new_i = (i - abs(nums[i])) % n
            else:
                new_i = i
            result.append(nums[new_i])
        return result