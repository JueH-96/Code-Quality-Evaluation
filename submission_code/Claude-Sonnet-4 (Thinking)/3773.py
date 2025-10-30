class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def is_non_decreasing(arr):
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True
        
        operations = 0
        
        while not is_non_decreasing(nums):
            # Find the adjacent pair with minimum sum (leftmost if tie)
            min_sum = float('inf')
            min_index = -1
            
            for i in range(len(nums) - 1):
                pair_sum = nums[i] + nums[i+1]
                if pair_sum < min_sum:
                    min_sum = pair_sum
                    min_index = i
            
            # Replace the pair with their sum
            new_value = nums[min_index] + nums[min_index + 1]
            nums[min_index] = new_value
            nums.pop(min_index + 1)
            operations += 1
        
        return operations