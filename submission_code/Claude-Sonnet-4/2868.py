class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        n = len(nums)
        left = 0
        result = 0
        
        # Deques to maintain indices of min and max elements in current window
        min_deque = deque()  # indices in increasing order of values
        max_deque = deque()  # indices in decreasing order of values
        
        for right in range(n):
            # Add current element to deques
            # Remove elements from back while current element is smaller (for min_deque)
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Remove elements from back while current element is larger (for max_deque)
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Shrink window from left while max - min > 2
            while min_deque and max_deque and nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # Remove elements that are out of window
                if min_deque[0] == left:
                    min_deque.popleft()
                if max_deque[0] == left:
                    max_deque.popleft()
                left += 1
            
            # All subarrays ending at 'right' and starting from 'left' to 'right' are valid
            result += right - left + 1
        
        return result