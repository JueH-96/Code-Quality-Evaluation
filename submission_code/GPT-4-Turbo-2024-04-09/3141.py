class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        from collections import deque
        
        n = len(nums)
        prefix_sum = [0] * (2 * n + 1)
        
        # Create a prefix sum array for twice the length of nums to simulate the infinite array
        for i in range(1, 2 * n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[(i - 1) % n]
        
        # Use deque to maintain the minimum length of subarray with sum equals to target
        min_length = float('inf')
        deque_index = deque()
        
        for i in range(2 * n + 1):
            # Remove elements from deque which are not useful
            while deque_index and prefix_sum[i] - prefix_sum[deque_index[0]] >= target:
                min_length = min(min_length, i - deque_index.popleft())
            
            # Maintain deque increasing
            while deque_index and prefix_sum[i] <= prefix_sum[deque_index[-1]]:
                deque_index.pop()
            
            deque_index.append(i)
        
        return min_length if min_length != float('inf') else -1