class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        n = len(nums)
        
        # Try all possible starting positions
        for start in range(n):
            current_sum = 0
            
            # Try all possible lengths from l to r
            for length in range(1, r + 1):
                # Check if we can form a subarray of this length starting at 'start'
                if start + length > n:
                    break
                
                # Add the next element to current sum
                current_sum += nums[start + length - 1]
                
                # If length is within our range [l, r] and sum is positive
                if length >= l and current_sum > 0:
                    min_sum = min(min_sum, current_sum)
        
        return min_sum if min_sum != float('inf') else -1