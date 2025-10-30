class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        operations = 0
        
        while True:
            # Check if the array is already non-decreasing
            is_non_decreasing = True
            for i in range(len(nums) - 1):
                if nums[i] > nums[i + 1]:
                    is_non_decreasing = False
                    break
            
            if is_non_decreasing:
                break
            
            # Find the adjacent pair with the minimum sum
            min_sum = float('inf')
            min_index = -1
            for i in range(len(nums) - 1):
                current_sum = nums[i] + nums[i + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i
            
            # Replace the pair with their sum
            nums[min_index] = min_sum
            nums.pop(min_index + 1)
            
            # Increment the operation count
            operations += 1
        
        return operations