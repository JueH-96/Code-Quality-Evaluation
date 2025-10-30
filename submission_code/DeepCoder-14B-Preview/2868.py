from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        left = 0
        max_deque = deque()
        min_deque = deque()
        result = 0
        
        for right in range(n):
            # Update max deque
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min deque
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Adjust the left pointer to maintain the condition
            while True:
                # Remove elements from deques that are out of the current window
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
                
                # If deques are empty, break (though this shouldn't happen as the window is non-empty)
                if not max_deque or not min_deque:
                    break
                
                current_max = nums[max_deque[0]]
                current_min = nums[min_deque[0]]
                
                if current_max - current_min > 2:
                    left += 1
                else:
                    break
            
            # Add the number of valid subarrays ending at 'right'
            result += (right - left + 1)
        
        return result