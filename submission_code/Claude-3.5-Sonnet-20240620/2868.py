class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        n = len(nums)
        result = 0
        left = 0
        max_queue = deque()
        min_queue = deque()
        
        for right in range(n):
            while max_queue and nums[max_queue[-1]] <= nums[right]:
                max_queue.pop()
            max_queue.append(right)
            
            while min_queue and nums[min_queue[-1]] >= nums[right]:
                min_queue.pop()
            min_queue.append(right)
            
            while nums[max_queue[0]] - nums[min_queue[0]] > 2:
                left += 1
                if max_queue[0] < left:
                    max_queue.popleft()
                if min_queue[0] < left:
                    min_queue.popleft()
            
            result += right - left + 1
        
        return result