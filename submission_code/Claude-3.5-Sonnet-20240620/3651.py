class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] == 0:
                result[i] = nums[i]
            else:
                steps = nums[i]
                if steps > 0:
                    target_index = (i + steps) % n
                else:
                    target_index = (i + steps) % n
                    if target_index < 0:
                        target_index += n
                result[i] = nums[target_index]
        
        return result