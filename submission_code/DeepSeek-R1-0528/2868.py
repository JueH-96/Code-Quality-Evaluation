from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        left = 0
        max_deque = deque()
        min_deque = deque()
        res = 0
        
        for right, num in enumerate(nums):
            while max_deque and nums[max_deque[-1]] < num:
                max_deque.pop()
            max_deque.append(right)
            
            while min_deque and nums[min_deque[-1]] > num:
                min_deque.pop()
            min_deque.append(right)
            
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                left += 1
            
            res += (right - left + 1)
        
        return res