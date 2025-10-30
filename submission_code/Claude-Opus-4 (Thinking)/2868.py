class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        n = len(nums)
        left = 0
        count = 0
        min_deque = deque()  # Stores indices, maintaining increasing order of values
        max_deque = deque()  # Stores indices, maintaining decreasing order of values
        
        for right in range(n):
            # Update min_deque - remove elements >= current from back
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Update max_deque - remove elements <= current from back
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Shrink window if max - min > 2
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove indices outside window
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
            
            # Add count of all subarrays ending at right
            count += right - left + 1
        
        return count