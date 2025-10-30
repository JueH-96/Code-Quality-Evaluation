class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        operations = 0
        
        while True:
            # Check if the array is already non-decreasing
            if all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)):
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
            nums[min_index] = nums[min_index] + nums[min_index + 1]
            del nums[min_index + 1]  # Remove the next element
            operations += 1
        
        return operations