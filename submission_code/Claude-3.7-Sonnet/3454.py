class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        
        operations = 0
        i = 0
        
        while i < n:
            # Skip elements that are already equal
            if diff[i] == 0:
                i += 1
                continue
            
            # Determine the sign of the current group
            sign = 1 if diff[i] > 0 else -1
            
            # Find the maximum absolute value in the current group
            max_abs = 0
            while i < n and diff[i] * sign > 0:
                max_abs = max(max_abs, abs(diff[i]))
                i += 1
            
            operations += max_abs
        
        return operations