class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Process from right to left
        # This allows us to greedily merge elements whenever possible
        n = len(nums)
        
        # Start from the second last element and go backwards
        for i in range(n - 2, -1, -1):
            # If current element is <= next element, we can merge
            if nums[i] <= nums[i + 1]:
                # Merge by adding current to next
                nums[i + 1] = nums[i] + nums[i + 1]
                # Mark current as 0 (effectively deleted)
                nums[i] = 0
        
        # Return the maximum element in the array
        return max(nums)