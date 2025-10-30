class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        current_length = 0
        start = 0
        
        while start < len(nums):
            # Find the first even number <= threshold to start a valid subarray
            while start < len(nums) and (nums[start] % 2 != 0 or nums[start] > threshold):
                start += 1
            
            if start >= len(nums):
                break
            
            # Initialize the current subarray
            current_length = 1
            i = start
            
            while i + 1 < len(nums):
                # Check if the next element alternates in parity and is within the threshold
                if (nums[i] % 2 != nums[i + 1] % 2) and (nums[i + 1] <= threshold):
                    current_length += 1
                    i += 1
                else:
                    break
            
            # Update the maximum length found
            max_length = max(max_length, current_length)
            
            # Move start to the next possible position
            start += 1
        
        return max_length