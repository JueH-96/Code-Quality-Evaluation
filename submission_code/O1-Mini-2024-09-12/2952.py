from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        a_sum = sum(nums1)
        b_sum = sum(nums2)
        if a_sum <= x:
            return 0
        sorted_indices = sorted(range(n), key=lambda i: nums2[i], reverse=True)
        prefix_a = [0] * (n + 1)
        prefix_b = [0] * (n + 1)
        prefix_bk = [0] * (n + 1)
        for i in range(n):
            idx = sorted_indices[i]
            prefix_a[i+1] = prefix_a[i] + nums1[idx]
            prefix_b[i+1] = prefix_b[i] + nums2[idx]
            prefix_bk[i+1] = prefix_bk[i] + nums2[idx] * i
        left, right = 1, n
        answer = -1
        while left <= right:
            mid = (left + right) // 2
            sum_set_zero = prefix_a[mid] + prefix_b[mid] * mid - prefix_bk[mid]
            required_sum = a_sum + mid * b_sum - sum_set_zero
            if required_sum <= x:
                answer = mid
                right = mid -1
            else:
                left = mid +1
        return answer