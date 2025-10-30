class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        shortest_length = float('inf')
        current_or = 0
        left = 0
        
        for right in range(len(nums)):
            current_or |= nums[right]
            while current_or >= k:
                shortest_length = min(shortest_length, right - left + 1)
                current_or ^= nums[left]
                left += 1
        
        return shortest_length if shortest_length != float('inf') else -1