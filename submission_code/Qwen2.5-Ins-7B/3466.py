class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        prefix_and = 0
        left = 0
        
        for right in range(len(nums)):
            prefix_and &= nums[right]
            while left <= right and prefix_and < k:
                prefix_and &= nums[left]
                left += 1
            if prefix_and == k:
                count += right - left + 1
        
        return count