class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if k not in nums and all(x < k for x in nums):
            return -1
        
        nums.sort()
        operations = 0
        current_max = nums[-1]
        
        while current_max != k:
            # Find the largest value less than current_max
            target = None
            for i in range(len(nums) - 1, -1, -1):
                if nums[i] < current_max:
                    target = nums[i]
                    break
            
            # If no target found, it means all are equal to current_max
            if target is None:
                if current_max == k:
                    return operations
                else:
                    return -1
            
            # Apply operation to reduce all values greater than target to target
            for i in range(len(nums)):
                if nums[i] > target:
                    nums[i] = target
            
            current_max = target
            operations += 1
        
        return operations