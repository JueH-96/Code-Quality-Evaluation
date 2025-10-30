class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        count = 0
        length = 1  # Length of current alternating sequence
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                # Continue the alternating sequence
                length += 1
            else:
                # Sequence breaks, add count of all subarrays in previous sequence
                count += length * (length + 1) // 2
                length = 1  # Start new sequence with current element
        
        # Don't forget to add subarrays from the last sequence
        count += length * (length + 1) // 2
        
        return count