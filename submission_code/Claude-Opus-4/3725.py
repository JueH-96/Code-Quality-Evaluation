class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import deque
        
        n = len(nums)
        total_sum = 0
        
        # For each subarray length from 1 to k
        for length in range(1, min(k + 1, n + 1)):
            # Deques to maintain indices of elements in current window
            # min_deque maintains indices in increasing order of values
            # max_deque maintains indices in decreasing order of values
            min_deque = deque()
            max_deque = deque()
            
            # Initialize window for first subarray of current length
            for i in range(length):
                # Maintain min_deque
                while min_deque and nums[min_deque[-1]] >= nums[i]:
                    min_deque.pop()
                min_deque.append(i)
                
                # Maintain max_deque
                while max_deque and nums[max_deque[-1]] <= nums[i]:
                    max_deque.pop()
                max_deque.append(i)
            
            # Add min + max of first window
            total_sum += nums[min_deque[0]] + nums[max_deque[0]]
            
            # Slide the window
            for i in range(length, n):
                # Remove elements outside window
                if min_deque and min_deque[0] <= i - length:
                    min_deque.popleft()
                if max_deque and max_deque[0] <= i - length:
                    max_deque.popleft()
                
                # Add new element
                while min_deque and nums[min_deque[-1]] >= nums[i]:
                    min_deque.pop()
                min_deque.append(i)
                
                while max_deque and nums[max_deque[-1]] <= nums[i]:
                    max_deque.pop()
                max_deque.append(i)
                
                # Add min + max of current window
                total_sum += nums[min_deque[0]] + nums[max_deque[0]]
        
        return total_sum