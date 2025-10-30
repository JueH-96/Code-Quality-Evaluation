class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        nums = [1]
        operations = 0
        
        while sum(nums) < k:
            max_val = max(nums)
            max_index = nums.index(max_val)
            
            increase_needed = k - sum(nums)
            
            if increase_needed >= max_val:
                operations += max_val
                nums[max_index] += max_val
            else:
                operations += increase_needed
                nums[max_index] += increase_needed
                break

        return operations