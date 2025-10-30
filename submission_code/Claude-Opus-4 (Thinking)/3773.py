class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        nums = nums[:]  # Make a copy to avoid modifying the input
        operations = 0
        
        while True:
            # Check if array is non-decreasing
            non_decreasing = True
            for i in range(1, len(nums)):
                if nums[i] < nums[i - 1]:
                    non_decreasing = False
                    break
            
            if non_decreasing:
                break
            
            # Find the adjacent pair with minimum sum
            min_sum = float('inf')
            min_index = -1
            
            for i in range(len(nums) - 1):
                pair_sum = nums[i] + nums[i + 1]
                if pair_sum < min_sum:
                    min_sum = pair_sum
                    min_index = i
            
            # Replace the pair with their sum
            nums[min_index] = min_sum
            nums.pop(min_index + 1)
            
            operations += 1
        
        return operations