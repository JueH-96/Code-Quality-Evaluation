class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # If array is empty or has only distinct elements, return 0
        if len(set(nums)) == len(nums):
            return 0
            
        operations = 0
        i = 0
        
        # Continue until we process all elements
        while i < len(nums):
            # If remaining elements are all distinct, break
            if len(set(nums[i:])) == len(nums[i:]):
                break
                
            # Remove 3 elements (or all remaining if less than 3)
            remove_count = min(3, len(nums) - i)
            i += remove_count
            operations += 1
            
            # If we've removed all elements, break
            if i >= len(nums):
                break
        
        return operations