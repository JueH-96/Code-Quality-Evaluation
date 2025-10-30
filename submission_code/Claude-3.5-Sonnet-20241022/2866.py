class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        
        for l in range(n):
            # Skip if first number is odd or greater than threshold
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
                
            r = l
            # Keep extending right pointer while conditions are met
            while r < n and nums[r] <= threshold:
                if r > l and nums[r] % 2 == nums[r-1] % 2:
                    break
                max_len = max(max_len, r - l + 1)
                r += 1
                
        return max_len