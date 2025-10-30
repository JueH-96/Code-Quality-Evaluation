class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        
        # Keep removing from the front until array became distinct or empty
        while len(nums) != len(set(nums)):
            # Remove up to 3 elements from the front
            to_remove = min(3, len(nums))
            nums = nums[to_remove:]
            operations += 1
            if not nums:  # An empty array is considered distinct
                break
        
        return operations