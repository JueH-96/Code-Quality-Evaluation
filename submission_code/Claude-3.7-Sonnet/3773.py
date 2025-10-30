class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        operations = 0
        
        while True:
            # Check if the array is already non-decreasing
            is_non_decreasing = True
            for i in range(1, len(nums)):
                if nums[i] < nums[i-1]:
                    is_non_decreasing = False
                    break
            
            # If array is non-decreasing, return current operation count
            if is_non_decreasing:
                return operations
            
            # Find the leftmost adjacent pair with minimum sum
            min_sum = float('inf')
            min_idx = -1
            
            for i in range(len(nums) - 1):
                curr_sum = nums[i] + nums[i+1]
                if curr_sum < min_sum:
                    min_sum = curr_sum
                    min_idx = i
            
            # Replace the pair with their sum
            nums[min_idx] = min_sum
            nums.pop(min_idx + 1)
            
            operations += 1