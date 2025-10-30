class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        # If array length is 1, it's already non-decreasing
        if len(nums) == 1:
            return 0
        
        def is_non_decreasing(arr):
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True
        
        def find_min_sum_pair(arr):
            min_sum = float('inf')
            min_idx = -1
            
            # Find leftmost adjacent pair with minimum sum
            for i in range(len(arr)-1):
                curr_sum = arr[i] + arr[i+1]
                if curr_sum < min_sum:
                    min_sum = curr_sum
                    min_idx = i
            
            return min_idx
        
        operations = 0
        nums = nums.copy()  # Create a copy to avoid modifying input
        
        while not is_non_decreasing(nums):
            # Find index of leftmost pair with minimum sum
            idx = find_min_sum_pair(nums)
            
            # Replace pair with their sum
            pair_sum = nums[idx] + nums[idx + 1]
            nums.pop(idx)  # Remove first element of pair
            nums[idx] = pair_sum  # Replace second element with sum
            
            operations += 1
            
            # If we've done too many operations, array might be impossible to make non-decreasing
            if operations > len(nums):
                return -1
        
        return operations