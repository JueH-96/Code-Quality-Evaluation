class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        distinct_elements = set()
        operations = 0
        i = 0
        
        while i < len(nums):
            if len(nums) - i >= 3:
                # Remove 3 elements
                chunk = nums[i:i+3]
                distinct_elements.update(chunk)
                i += 3
            else:
                # Remove remaining elements
                chunk = nums[i:]
                distinct_elements.update(chunk)
                i = len(nums)
            
            operations += 1
            
            # Check if all elements are distinct
            if len(distinct_elements) == len(nums):
                return operations - 1  # Subtract 1 as the last operation was unnecessary
        
        return operations