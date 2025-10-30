from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # Step 1: Find the dominant element in the entire array.
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        dominant = None
        for num, count in freq.items():
            if count * 2 > n:  # dominant condition for whole nums
                dominant = num
                break

        # Step 2: Iterate to find a valid split index.
        total_count = freq[dominant]
        left_count = 0
        
        # We need to check for each split index i where 0 <= i < n - 1.
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            # For the left subarray: length = i+1
            # Check if dominant is dominant in left: left_count * 2 > (i + 1)
            if left_count * 2 <= (i + 1):
                continue
            
            # For the right subarray: length = n - i - 1, frequency = total_count - left_count
            if (total_count - left_count) * 2 > (n - i - 1):
                return i
        
        return -1