class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_len = float('inf')
        start = 0
        current_or = 0
        
        for end in range(len(nums)):
            current_or |= nums[end]
            
            while current_or >= k:
                min_len = min(min_len, end - start + 1)
                current_or ^= nums[start]
                start += 1
        
        return min_len if min_len != float('inf') else -1