class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i, v in enumerate(nums):
            if v > 0:
                steps = v % n
                idx = (i + steps) % n
                result[i] = nums[idx]
            elif v < 0:
                steps = (-v) % n
                idx = (i - steps) % n
                result[i] = nums[idx]
            else:
                result[i] = v
        
        return result