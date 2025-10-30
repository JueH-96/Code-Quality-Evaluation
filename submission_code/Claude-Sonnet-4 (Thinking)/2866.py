class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        
        for l in range(n):
            # Check if starting element is even and within threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                # Try to extend as far as possible
                r = l
                while r < n and nums[r] <= threshold and (r == l or nums[r] % 2 != nums[r-1] % 2):
                    r += 1
                
                max_length = max(max_length, r - l)
        
        return max_length