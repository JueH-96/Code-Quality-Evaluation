from collections import deque

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        total = 0
        
        for m in range(1, k + 1):
            min_deque = deque()
            max_deque = deque()
            current_sum = 0
            
            # Process the first window
            for i in range(m):
                while min_deque and nums[i] <= nums[min_deque[-1]]:
                    min_deque.pop()
                min_deque.append(i)
                
                while max_deque and nums[i] >= nums[max_deque[-1]]:
                    max_deque.pop()
                max_deque.append(i)
            
            current_sum += nums[min_deque[0]] + nums[max_deque[0]]
            
            # Process the rest of the windows
            for i in range(m, len(nums)):
                # Remove elements not in the current window
                while min_deque and min_deque[0] <= i - m:
                    min_deque.popleft()
                while max_deque and max_deque[0] <= i - m:
                    max_deque.popleft()
                
                # Add the new element
                while min_deque and nums[i] <= nums[min_deque[-1]]:
                    min_deque.pop()
                min_deque.append(i)
                
                while max_deque and nums[i] >= nums[max_deque[-1]]:
                    max_deque.pop()
                max_deque.append(i)
                
                current_sum += nums[min_deque[0]] + nums[max_deque[0]]
            
            total += current_sum
        
        return total