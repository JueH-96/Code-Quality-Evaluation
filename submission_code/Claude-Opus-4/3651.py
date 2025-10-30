class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n):
            if nums[i] > 0:
                # Move right by nums[i] steps
                new_index = (i + nums[i]) % n
                result.append(nums[new_index])
            elif nums[i] < 0:
                # Move left by abs(nums[i]) steps
                new_index = (i - abs(nums[i])) % n
                result.append(nums[new_index])
            else:
                # nums[i] == 0, stay at same position
                result.append(nums[i])
        
        return result