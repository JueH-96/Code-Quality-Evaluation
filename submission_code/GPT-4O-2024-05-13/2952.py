from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        
        if total_sum <= x:
            return 0
        
        # Create a list of tuples (nums1[i], nums2[i]) and sort by nums2[i] in descending order
        combined = sorted(zip(nums1, nums2), key=lambda p: p[1], reverse=True)
        
        # Initialize the current sum and the time counter
        current_sum = total_sum
        time = 0
        
        for i in range(n):
            if current_sum <= x:
                return time
            current_sum -= combined[i][0]
            time += 1
            for j in range(n):
                combined[j] = (combined[j][0] + combined[j][1], combined[j][1])
        
        return -1 if current_sum > x else time