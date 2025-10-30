class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Determine the length of the longest sequential prefix
        max_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                max_len += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        s = sum(nums[:max_len])
        
        # Create a set of the elements for quick lookup
        elements = set(nums)
        
        # Find the smallest x >= s that's not in the elements
        x = s
        while x in elements:
            x += 1
        
        return x