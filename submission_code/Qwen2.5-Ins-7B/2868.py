from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        left = 0
        min_deque = deque()
        max_deque = deque()
        result = 0
        
        for right in range(len(nums)):
            # Maintain the min_deque
            while min_deque and nums[min_deque[-1]] > nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Maintain the max_deque
            while max_deque and nums[max_deque[-1]] < nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Check the condition for the current window
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                left += 1
            
            result += right - left + 1
        
        return result