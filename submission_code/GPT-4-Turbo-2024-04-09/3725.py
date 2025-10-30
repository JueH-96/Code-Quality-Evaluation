class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import deque
        
        def maxSlidingWindow(nums, k):
            # This function returns the maximum values in each sliding window of size k
            n = len(nums)
            if n * k == 0:
                return []
            if k == 1:
                return nums
            
            def clean_deque(i):
                # Remove indexes of elements not from sliding window
                if deq and deq[0] == i - k:
                    deq.popleft()
                
                # Remove from deq indexes of all elements 
                # which are smaller than current element nums[i]
                while deq and nums[i] > nums[deq[-1]]:
                    deq.pop()
            
            # Initialize deque and output
            deq = deque()
            max_idx = 0
            for i in range(k):
                clean_deque(i)
                deq.append(i)
                # Compute max in nums[:k]
                if nums[i] > nums[max_idx]:
                    max_idx = i
            output = [nums[max_idx]]
            
            # Build output
            for i in range(k, n):
                clean_deque(i)
                deq.append(i)
                output.append(nums[deq[0]])
            return output
        
        def minSlidingWindow(nums, k):
            # This function returns the minimum values in each sliding window of size k
            n = len(nums)
            if n * k == 0:
                return []
            if k == 1:
                return nums
            
            nums = [-num for num in nums]  # Invert numbers to reuse max sliding window
            max_result = maxSlidingWindow(nums, k)
            return [-num for num in max_result]  # Revert to original values
        
        total_sum = 0
        for window_size in range(1, k + 1):
            max_values = maxSlidingWindow(nums, window_size)
            min_values = minSlidingWindow(nums, window_size)
            total_sum += sum(max_values) + sum(min_values)
        
        return total_sum