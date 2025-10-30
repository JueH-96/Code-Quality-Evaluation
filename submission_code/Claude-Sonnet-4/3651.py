class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n):
            if nums[i] == 0:
                result.append(nums[i])
            elif nums[i] > 0:
                # Move right
                new_index = (i + nums[i]) % n
                result.append(nums[new_index])
            else:
                # Move left
                new_index = (i + nums[i]) % n
                result.append(nums[new_index])
        
        return result