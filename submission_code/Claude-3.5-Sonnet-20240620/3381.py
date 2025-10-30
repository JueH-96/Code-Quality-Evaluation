class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = 0
        current_or = 0
        min_length = float('inf')
        
        for right in range(n):
            current_or |= nums[right]
            
            while current_or >= k:
                min_length = min(min_length, right - left + 1)
                current_or ^= nums[left]
                left += 1
                
                if left > right:
                    break
        
        return min_length if min_length != float('inf') else -1