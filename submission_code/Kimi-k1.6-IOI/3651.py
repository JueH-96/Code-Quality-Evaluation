class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n):
            if nums[i] == 0:
                result.append(0)
            else:
                if nums[i] > 0:
                    new_idx = (i + nums[i]) % n
                else:
                    new_idx = (i - abs(nums[i])) % n
                result.append(nums[new_idx])
        return result