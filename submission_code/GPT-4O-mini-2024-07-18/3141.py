class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        from collections import deque
        
        n = len(nums)
        prefix_sum = [0] * (2 * n + 1)
        
        # Create the prefix sum array for the first two cycles of nums
        for i in range(2 * n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i % n]
        
        min_length = float('inf')
        dq = deque()
        
        for i in range(2 * n + 1):
            # Maintain the deque to find the minimum length
            while dq and prefix_sum[i] - prefix_sum[dq[0]] >= target:
                min_length = min(min_length, i - dq.popleft())
            # Maintain the increasing order of prefix sums in the deque
            while dq and prefix_sum[i] <= prefix_sum[dq[-1]]:
                dq.pop()
            dq.append(i)
        
        return min_length if min_length != float('inf') else -1