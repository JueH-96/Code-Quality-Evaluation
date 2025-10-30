from collections import deque

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        total_sum = 0
        n = len(nums)
        
        for window_size in range(1, k + 1):
            # Using deques to track min and max elements in the current window
            min_deque = deque()
            max_deque = deque()
            
            for i in range(n):
                # Remove elements outside the current window
                while min_deque and min_deque[0] <= i - window_size:
                    min_deque.popleft()
                while max_deque and max_deque[0] <= i - window_size:
                    max_deque.popleft()
                
                # Remove elements that won't be useful for min/max computation
                while min_deque and nums[min_deque[-1]] >= nums[i]:
                    min_deque.pop()
                while max_deque and nums[max_deque[-1]] <= nums[i]:
                    max_deque.pop()
                
                min_deque.append(i)
                max_deque.append(i)
                
                # If we have a full window, add min+max to our result
                if i >= window_size - 1:
                    total_sum += nums[min_deque[0]] + nums[max_deque[0]]
        
        return total_sum