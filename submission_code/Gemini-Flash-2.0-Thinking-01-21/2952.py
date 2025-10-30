from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        indices = sorted(range(n), key=lambda i: nums2[i], reverse=True)
        
        def check_time(time):
            min_sum_at_time = float('inf')
            for k in range(min(n, time) + 1):
                current_sum = 0
                reset_indices = indices[:k]
                non_reset_indices = indices[k:]
                for j in range(k):
                    current_sum += (j) * nums2[reset_indices[j]]
                for j in range(len(non_reset_indices)):
                    current_sum += nums1[non_reset_indices[j]] + time * nums2[non_reset_indices[j]]
                min_sum_at_time = min(min_sum_at_time, current_sum)
            return min_sum_at_time <= x
            
        if sum(nums1) <= x:
            return 0
            
        low = 1
        high = 2001 # A sufficiently large upper bound, could be adjusted.
        min_time = -1
        
        while low <= high:
            mid_time = (low + high) // 2
            if check_time(mid_time):
                min_time = mid_time
                high = mid_time - 1
            else:
                low = mid_time + 1
                
        return min_time