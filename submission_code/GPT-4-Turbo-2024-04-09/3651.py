class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] == 0:
                result[i] = nums[i]
            else:
                steps = abs(nums[i])
                if nums[i] > 0:
                    new_index = (i + steps) % n
                else:
                    new_index = (i - steps) % n
                result[i] = nums[new_index]
        
        return result