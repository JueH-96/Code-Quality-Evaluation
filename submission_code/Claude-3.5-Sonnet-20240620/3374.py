class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0
        
        while i < n:
            # Count single-element subarrays
            count += 1
            
            # Check for alternating subarrays starting from current index
            j = i + 1
            while j < n and nums[j] != nums[j-1]:
                count += j - i + 1
                j += 1
            
            # Move to the next different element
            i = j
        
        return count