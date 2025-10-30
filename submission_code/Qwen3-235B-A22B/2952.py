from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        indices = list(range(n))
        # Sort elements based on nums2 descending, then nums1 descending to break ties
        indices.sort(key=lambda i: (-nums2[i], -nums1[i]))
        sorted_nums1 = [nums1[i] for i in indices]
        sorted_nums2 = [nums2[i] for i in indices]
        
        # Precompute prefix sums for nums1, nums2, and nums2[i] * i
        prefix_nums1 = [0] * (n + 1)
        prefix_nums2 = [0] * (n + 1)
        prefix_c = [0] * (n + 1)
        for i in range(n):
            prefix_nums1[i+1] = prefix_nums1[i] + sorted_nums1[i]
            prefix_nums2[i+1] = prefix_nums2[i] + sorted_nums2[i]
            prefix_c[i+1] = prefix_c[i] + sorted_nums2[i] * i
        
        sum_nums1 = prefix_nums1[n]
        sum_nums2 = prefix_nums2[n]
        
        # Handle the special case when all nums2[i] are zero
        if sum_nums2 == 0:
            if sum_nums1 <= x:
                return 0
            needed = sum_nums1 - x
            # Find the smallest k such that prefix_nums1[k] >= needed
            for k in range(1, n+1):
                if prefix_nums1[k] >= needed:
                    return k
            return -1
        
        # Try increasing values of t up to a reasonable upper bound
        # We use 5000 as the upper bound here; can be adjusted for performance
        for t in range(0, 5000):
            required = sum_nums1 + t * sum_nums2 - x
            if required <= 0:
                return t
            max_k = min(t, n)
            # Check for each possible k (number of elements reset)
            for k in range(1, max_k + 1):
                current_sum_resets = prefix_nums1[k] + t * prefix_nums2[k] - prefix_c[k]
                if current_sum_resets >= required:
                    return t
            # Early exit optimization: if sorted_nums2[0] is zero, further t won't help
            if sorted_nums2[0] == 0:
                break
        
        # Check if the global condition (using all elements) allows solution
        if prefix_c[n] <= x:
            return n
        return -1