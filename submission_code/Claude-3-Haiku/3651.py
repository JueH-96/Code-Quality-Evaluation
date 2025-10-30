class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] > 0:
                new_index = (i + nums[i]) % n
                result[new_index] = i
            elif nums[i] < 0:
                new_index = (i - abs(nums[i])) % n
                result[new_index] = i
            else:
                result[i] = 0
        
        return result