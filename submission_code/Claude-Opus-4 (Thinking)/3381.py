class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        if k == 0:
            return 1
        
        n = len(nums)
        min_length = float('inf')
        
        # Try all possible starting positions
        for i in range(n):
            or_val = 0
            # Expand subarray from position i
            for j in range(i, n):
                or_val |= nums[j]
                if or_val >= k:
                    min_length = min(min_length, j - i + 1)
                    break  # No need to expand further from position i
        
        return min_length if min_length != float('inf') else -1