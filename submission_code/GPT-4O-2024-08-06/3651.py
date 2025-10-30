class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] > 0:
                # Move nums[i] steps to the right
                new_index = (i + nums[i]) % n
                result[i] = nums[new_index]
            elif nums[i] < 0:
                # Move abs(nums[i]) steps to the left
                new_index = (i + nums[i]) % n
                result[i] = nums[new_index]
            else:
                # nums[i] == 0
                result[i] = nums[i]
        
        return result