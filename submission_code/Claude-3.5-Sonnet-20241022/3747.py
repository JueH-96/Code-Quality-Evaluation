class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = 0
        
        # Check all adjacent pairs including the wrap-around pair
        for i in range(n):
            # Get current element and next element (using modulo for wrap-around)
            curr = nums[i]
            next_elem = nums[(i + 1) % n]
            
            # Calculate absolute difference
            diff = abs(curr - next_elem)
            
            # Update maximum difference if current difference is larger
            max_diff = max(max_diff, diff)
            
        return max_diff