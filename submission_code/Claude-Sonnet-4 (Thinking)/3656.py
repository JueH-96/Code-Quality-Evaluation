class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        for operations in range((n + 2) // 3 + 1):
            removed = 3 * operations
            if removed >= n:
                return operations
            
            remaining = nums[removed:]
            if len(set(remaining)) == len(remaining):
                return operations