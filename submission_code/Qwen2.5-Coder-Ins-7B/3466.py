class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        left = 0
        current_and = 0
        
        for right in range(len(nums)):
            current_and &= nums[right]
            while current_and < k and left <= right:
                current_and |= nums[left]
                left += 1
            if current_and == k:
                count += right - left + 1
        
        return count