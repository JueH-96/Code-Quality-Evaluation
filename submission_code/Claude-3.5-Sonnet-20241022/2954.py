class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        n = len(nums)
        
        # Sliding window approach
        for i in range(n - k + 1):
            # Get current window
            window = nums[i:i+k]
            # Check if window has at least m distinct elements
            if len(set(window)) >= m:
                # Update max_sum if current window sum is larger
                max_sum = max(max_sum, sum(window))
                
        return max_sum