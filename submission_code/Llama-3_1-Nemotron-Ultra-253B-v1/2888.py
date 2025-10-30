from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element using Boyer-Moore Majority Vote Algorithm
        d = None
        count = 0
        for num in nums:
            if count == 0:
                d = num
                count = 1
            else:
                count += 1 if num == d else -1
        
        # Calculate total occurrences of the dominant element
        total_d = sum(1 for num in nums if num == d)
        
        current_count = 0
        for i in range(len(nums)):
            if nums[i] == d:
                current_count += 1
            
            # Check if current index i is a valid split
            if i < len(nums) - 1:
                left_ok = current_count * 2 > (i + 1)
                right_count = total_d - current_count
                right_length = len(nums) - (i + 1)
                right_ok = right_count * 2 > right_length
                
                if left_ok and right_ok:
                    return i
        
        return -1