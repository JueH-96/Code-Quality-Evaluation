class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] == 0:
                # If value is 0, set result[i] to 0
                result[i] = 0
            else:
                # Calculate the landing index using modulo for circular behavior
                landing_index = (i + nums[i]) % n
                # Set result[i] to the value at the landing index
                result[i] = nums[landing_index]
        
        return result