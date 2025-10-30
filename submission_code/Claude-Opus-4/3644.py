class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        # Try all possible subarray lengths from l to r
        for length in range(l, r + 1):
            # Calculate sum of first window of current length
            window_sum = sum(nums[:length])
            
            # Check if this sum is positive
            if window_sum > 0:
                min_sum = min(min_sum, window_sum)
                found = True
            
            # Slide the window through the rest of the array
            for i in range(length, n):
                # Remove the leftmost element and add the new rightmost element
                window_sum = window_sum - nums[i - length] + nums[i]
                
                # Check if this sum is positive
                if window_sum > 0:
                    min_sum = min(min_sum, window_sum)
                    found = True
        
        return min_sum if found else -1