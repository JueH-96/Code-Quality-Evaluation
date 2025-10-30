class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def is_non_decreasing(arr):
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True
        
        operations = 0
        
        while not is_non_decreasing(nums):
            # Find the minimum sum pair (leftmost if tie)
            min_sum = float('inf')
            min_index = -1
            
            for i in range(len(nums) - 1):
                pair_sum = nums[i] + nums[i + 1]
                if pair_sum < min_sum:
                    min_sum = pair_sum
                    min_index = i
            
            # Replace the pair with their sum
            new_nums = []
            for i in range(len(nums)):
                if i == min_index:
                    new_nums.append(nums[i] + nums[i + 1])
                elif i == min_index + 1:
                    continue  # Skip the second element of the pair
                else:
                    new_nums.append(nums[i])
            
            nums = new_nums
            operations += 1
        
        return operations