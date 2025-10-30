class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        left, right = 0, 0
        min_diff = float('inf')
        current_or = 0
        
        while right < len(nums):
            current_or |= nums[right]
            while left <= right and current_or - k >= 0:
                min_diff = min(min_diff, abs(current_or - k))
                current_or &= ~nums[left]
                left += 1
            right += 1
        
        return min_diff if min_diff != float('inf') else abs(current_or - k)