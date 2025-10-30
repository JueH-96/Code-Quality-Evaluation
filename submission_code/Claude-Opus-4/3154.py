class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = 0  # Maximum triplet value found
        max_diff = 0   # Maximum value of (nums[i] - nums[j]) where i < j
        max_num = 0    # Maximum number seen so far
        
        for num in nums:
            # For current position k, calculate (nums[i] - nums[j]) * nums[k]
            # using the best (nums[i] - nums[j]) found so far
            max_value = max(max_value, max_diff * num)
            
            # Update max_diff for future k positions
            # This represents the best (nums[i] - nums[j]) where j is current position
            max_diff = max(max_diff, max_num - num)
            
            # Update max_num for future j positions
            max_num = max(max_num, num)
        
        return max_value