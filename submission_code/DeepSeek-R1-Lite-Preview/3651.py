class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n):
            if nums[i] > 0:
                landing_index = (i + nums[i]) % n
            elif nums[i] < 0:
                landing_index = (i - abs(nums[i])) % n
            else:
                landing_index = i
            result.append(nums[landing_index])
        return result