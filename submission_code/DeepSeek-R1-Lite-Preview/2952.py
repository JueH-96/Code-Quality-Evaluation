from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Binary search over time t
        left, right = 0, 10**6  # Upper bound can be adjusted based on constraints
        answer = -1
        
        while left <= right:
            mid = (left + right) // 2
            # Calculate the sum if we set the t largest (nums1[i] + mid * nums2[i]) to zero
            final_values = [nums1[i] + mid * nums2[i] for i in range(n)]
            final_values.sort()
            # Sum of the smallest n - mid elements
            if mid >= n:
                current_sum = 0
            else:
                current_sum = sum(final_values[:n - mid])
            if current_sum <= x:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer