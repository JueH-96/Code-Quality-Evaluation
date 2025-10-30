class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_val = 0  # Maximum triplet value found
        max_i = 0    # Maximum nums[i] seen so far
        max_diff = 0 # Maximum (nums[i] - nums[j]) seen so far
        
        for k in range(len(nums)):
            # Update max_val using current nums[k] as the third element
            max_val = max(max_val, max_diff * nums[k])
            
            # Update max_diff using current nums[k] as nums[j]
            max_diff = max(max_diff, max_i - nums[k])
            
            # Update max_i for future iterations
            max_i = max(max_i, nums[k])
        
        return max_val