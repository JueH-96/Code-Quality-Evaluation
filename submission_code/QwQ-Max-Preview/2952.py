from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        sum_initial = sum(nums1)
        sumB = sum(nums2)
        n = len(nums1)
        
        if sum_initial <= x:
            return 0
        
        if sumB == 0:
            sorted_nums1 = sorted(nums1, reverse=True)
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] + sorted_nums1[i]
                if sum_initial - prefix[i+1] <= x:
                    return i+1
            return -1
        
        sorted_nums2 = sorted(nums2, reverse=True)
        left, right = 1, 2 * 10**5  # Adjust based on constraints, but 1e5 is safe
        
        def is_possible(t):
            current_time = t
            sum_a = 0
            remaining = t
            for num in sorted_nums2:
                if remaining <= 0:
                    break
                s_i = min(remaining, current_time)
                sum_times = s_i * (2 * current_time - s_i + 1) // 2
                sum_a += num * sum_times
                remaining -= s_i
                current_time -= s_i
            return sum_initial + sumB * t - sum_a <= x
        
        answer = -1
        left = 1
        right = 0
        while True:
            current_max = sum_initial + sumB * right
            if current_max < x:
                break
            right += 1
        right = max(right, 1)
        
        for t in range(1, right + 1):
            if is_possible(t):
                return t
        
        return -1