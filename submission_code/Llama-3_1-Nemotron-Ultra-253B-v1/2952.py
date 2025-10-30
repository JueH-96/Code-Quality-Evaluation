from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Sort the elements by nums2 descending, then nums1 descending
        sorted_pairs = sorted(zip(nums2, nums1), key=lambda x: (-x[0], -x[1]))
        b_sorted = [x[0] for x in sorted_pairs]
        a_sorted = [x[1] for x in sorted_pairs]
        
        # Precompute prefix sums for a and b
        prefix_a = [0] * (n + 1)
        prefix_b = [0] * (n + 1)
        for i in range(n):
            prefix_a[i + 1] = prefix_a[i] + a_sorted[i]
            prefix_b[i + 1] = prefix_b[i] + b_sorted[i]
        
        low = 0
        high = 10**18
        answer = -1
        
        def compute_sum_i_b(k):
            res = 0
            for i in range(k):
                res += i * b_sorted[i]
            return res
        
        while low <= high:
            mid = (low + high) // 2
            k = min(mid, n)
            sum_i_b = compute_sum_i_b(k)
            sum_a_not = prefix_a[n] - prefix_a[k]
            sum_b_not = prefix_b[n] - prefix_b[k]
            total = sum_a_not + mid * sum_b_not + sum_i_b
            if total <= x:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer if answer != -1 else -1