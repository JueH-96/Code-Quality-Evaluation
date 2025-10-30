class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find the maximum number in the array
        max_num = max(nums)
        
        # Calculate the expected length of the base array
        expected_length = max_num + 1
        
        # Check if the length of nums matches the expected length
        if len(nums) != expected_length:
            return False
        
        # Create the base array for comparison
        base = list(range(1, max_num)) + [max_num, max_num]
        
        # Sort both arrays and compare
        return sorted(nums) == sorted(base)