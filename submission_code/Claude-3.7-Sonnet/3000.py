class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
        
        n = len(nums)
        min_diff = float('inf')
        
        # We'll use a sorted list to keep track of values we've seen so far
        # that are at least x positions away from the current position
        left_values = []
        
        for i in range(x, n):
            # Add the element that's exactly x positions behind current position
            bisect.insort(left_values, nums[i - x])
            
            # Find the insertion point of current element in our sorted list
            idx = bisect.bisect_left(left_values, nums[i])
            
            # Check the element at or right after the insertion point
            if idx < len(left_values):
                min_diff = min(min_diff, abs(nums[i] - left_values[idx]))
            
            # Check the element right before the insertion point
            if idx > 0:
                min_diff = min(min_diff, abs(nums[i] - left_values[idx - 1]))
        
        return min_diff