class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_length = float('inf')
        
        # Try all possible subarrays
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                
                # If current OR is at least k, we found a valid subarray
                if current_or >= k:
                    min_length = min(min_length, j - i + 1)
                    break  # No need to extend further from this starting point
        
        return min_length if min_length != float('inf') else -1