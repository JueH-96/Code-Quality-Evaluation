class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        min_q = deque()  # will store indices in increasing order of nums values
        max_q = deque()  # will store indices in decreasing order of nums values
        
        left = 0
        result = 0
        
        for right, val in enumerate(nums):
            # Maintain max_q (front is largest)
            while max_q and nums[max_q[-1]] < val:
                max_q.pop()
            max_q.append(right)
            
            # Maintain min_q (front is smallest)
            while min_q and nums[min_q[-1]] > val:
                min_q.pop()
            min_q.append(right)
            
            # Shrink window while invalid (difference > 2)
            while nums[max_q[0]] - nums[min_q[0]] > 2:
                # Move left pointer
                if max_q and max_q[0] == left:
                    max_q.popleft()
                if min_q and min_q[0] == left:
                    min_q.popleft()
                left += 1
            
            # All subarrays ending at right with start from left..right are valid
            result += (right - left + 1)
        
        return result