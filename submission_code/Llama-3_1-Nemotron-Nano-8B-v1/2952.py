from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        initial_sum = sum(nums1)
        if initial_sum <= x:
            return 0
        
        sum_nums2 = sum(nums2)
        if sum_nums2 == 0:
            # Handle case where sum_nums2 is zero
            nums1_sorted = sorted(nums1, reverse=True)
            target = initial_sum - x
            current_sum = 0
            res = -1
            for i in range(n):
                current_sum += nums1_sorted[i]
                if current_sum >= target:
                    res = i + 1
                    break
            if res != -1:
                return res
            else:
                # Check if after all elements are reset, sum is <=x
                if x >= 0:
                    return n
                else:
                    return -1
        else:
            # Binary search for sum_nums2 > 0
            left, right = 0, 1 << 60
            answer = -1
            while left <= right:
                mid = (left + right) // 2
                t = min(mid, n)
                # Calculate the values for each element at mid steps
                elements = [nums1[i] + mid * nums2[i] for i in range(n)]
                elements.sort(reverse=True)
                sum_resets = sum(elements[:t])
                total = initial_sum + mid * sum_nums2 - sum_resets
                if total <= x:
                    answer = mid
                    right = mid - 1
                else:
                    left = mid + 1
            return answer if answer != -1 else -1