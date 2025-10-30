class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        # For each possible subarray length from 1 to k
        for length in range(1, k + 1):
            # Use sliding window to process all subarrays of current length
            window = deque()
            min_window = deque()
            max_window = deque()
            
            # Process first window
            for i in range(length):
                # Maintain min deque
                while min_window and nums[min_window[-1]] > nums[i]:
                    min_window.pop()
                min_window.append(i)
                
                # Maintain max deque
                while max_window and nums[max_window[-1]] < nums[i]:
                    max_window.pop()
                max_window.append(i)
            
            # Add sum for first window
            total += nums[min_window[0]] + nums[max_window[0]]
            
            # Process remaining windows
            for i in range(length, n):
                # Remove elements outside current window
                while min_window and min_window[0] <= i - length:
                    min_window.popleft()
                while max_window and max_window[0] <= i - length:
                    max_window.popleft()
                
                # Add current element
                while min_window and nums[min_window[-1]] > nums[i]:
                    min_window.pop()
                min_window.append(i)
                
                while max_window and nums[max_window[-1]] < nums[i]:
                    max_window.pop()
                max_window.append(i)
                
                # Add sum for current window
                total += nums[min_window[0]] + nums[max_window[0]]
        
        return total