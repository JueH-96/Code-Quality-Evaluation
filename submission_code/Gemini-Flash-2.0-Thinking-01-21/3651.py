class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            val = nums[i]
            if val > 0:
                landing_index = (i + val) % n
                result[i] = nums[landing_index]
            elif val < 0:
                landing_index = (i - abs(val)) % n
                result[i] = nums[landing_index]
            else:
                result[i] = nums[i]
        return result