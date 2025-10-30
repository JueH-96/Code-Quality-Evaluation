import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        # For each index i, maintain available values from left side
        left_available = []
        
        for i in range(n):
            # Add values that become available from the left
            if i >= x:
                bisect.insort(left_available, nums[i - x])
            
            # Check against left available values
            if left_available:
                pos = bisect.bisect_left(left_available, nums[i])
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - left_available[pos - 1]))
                if pos < len(left_available):
                    min_diff = min(min_diff, abs(nums[i] - left_available[pos]))
        
        # For each index i, maintain available values from right side
        right_available = []
        
        for i in range(n - 1, -1, -1):
            # Add values that become available from the right
            if i + x < n:
                bisect.insort(right_available, nums[i + x])
            
            # Check against right available values
            if right_available:
                pos = bisect.bisect_left(right_available, nums[i])
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - right_available[pos - 1]))
                if pos < len(right_available):
                    min_diff = min(min_diff, abs(nums[i] - right_available[pos]))
        
        return min_diff