class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        
        for l in range(n):
            if nums[l] % 2 == 0:
                r = l
                while r < n and nums[r] <= threshold:
                    if r == n - 1 or (nums[r] % 2 != nums[r + 1] % 2):
                        max_length = max(max_length, r - l + 1)
                        break
                    r += 1
        
        return max_length