class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        
        for l in range(n):
            # Check if starting element is even and within threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                r = l
                # Extend as far as possible
                while r < n and nums[r] <= threshold:
                    # Check alternating condition for elements after the first
                    if r > l and nums[r] % 2 == nums[r-1] % 2:
                        break
                    r += 1
                
                # Update max length
                max_length = max(max_length, r - l)
        
        return max_length