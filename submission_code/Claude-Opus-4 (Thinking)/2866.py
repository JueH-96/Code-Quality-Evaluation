class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        
        for i in range(n):
            # Check if we can start a subarray at position i
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                # Try to extend the subarray
                j = i
                
                # Keep extending while conditions are met
                while j + 1 < n and nums[j + 1] <= threshold and nums[j] % 2 != nums[j + 1] % 2:
                    j += 1
                
                # Update max_length
                max_length = max(max_length, j - i + 1)
        
        return max_length