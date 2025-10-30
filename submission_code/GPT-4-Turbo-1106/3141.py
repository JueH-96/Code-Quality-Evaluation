from collections import deque

class Solution:
    def minSizeSubarray(self, nums: list[int], target: int) -> int:
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        # To handle the case where the subarray might wrap around the end of the array,
        # we extend the prefix_sum array by appending the array to itself minus the last element.
        prefix_sum += [x + prefix_sum[-1] for x in prefix_sum[1:-1]]
        
        result = float('inf')
        window = deque()
        
        for i, total in enumerate(prefix_sum):
            # Maintain a monotonically increasing deque
            while window and total <= prefix_sum[window[-1]]:
                window.pop()
            # Check if the current window sum meets or exceeds the target
            while window and total - prefix_sum[window[0]] >= target:
                result = min(result, i - window.popleft())
            window.append(i)
        
        return result if result != float('inf') else -1