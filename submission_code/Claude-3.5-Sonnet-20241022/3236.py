class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find length of longest sequential prefix
        prefix_len = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1] + 1:
                break
            prefix_len += 1
            
        # Calculate sum of sequential prefix
        prefix_sum = sum(nums[:prefix_len])
        
        # Find smallest missing number >= prefix_sum
        x = prefix_sum
        while x in nums:
            x += 1
            
        return x