class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        n = len(nums)
        left = 0
        count = 0
        
        # Deques to maintain min and max in current window
        min_deque = deque()  # Increasing order (front has minimum)
        max_deque = deque()  # Decreasing order (front has maximum)
        
        for right in range(n):
            # Add current element to deques
            # For min_deque, remove elements larger than current from back
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # For max_deque, remove elements smaller than current from back
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Shrink window while condition is violated
            while min_deque and max_deque and nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove indices that are out of window
                if min_deque[0] < left:
                    min_deque.popleft()
                if max_deque[0] < left:
                    max_deque.popleft()
            
            # All subarrays ending at right and starting from left to right are valid
            count += right - left + 1
        
        return count