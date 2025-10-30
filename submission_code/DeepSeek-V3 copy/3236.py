class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 1
        
        # Find the longest sequential prefix
        max_length = 1
        current_length = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        prefix_sum = sum(nums[:max_length])
        
        # Find the smallest missing integer >= prefix_sum
        missing = prefix_sum
        while missing in nums:
            missing += 1
        
        return missing