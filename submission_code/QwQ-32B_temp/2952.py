class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Sort the pairs in descending order of nums2[i]
        pairs = sorted(zip(nums1, nums2), key=lambda p: (-p[1], -p[0]))
        nums1_sorted = [p[0] for p in pairs]
        nums2_sorted = [p[1] for p in pairs]
        
        # Compute prefix arrays
        prefix_nums1 = [0] * (n + 1)
        prefix_nums2 = [0] * (n + 1)
        prefix_terms = [0] * (n + 1)
        
        for i in range(1, n + 1):
            prefix_nums1[i] = prefix_nums1[i-1] + nums1_sorted[i-1]
            prefix_nums2[i] = prefix_nums2[i-1] + nums2_sorted[i-1]
            prefix_terms[i] = prefix_terms[i-1] + (i-1) * nums2_sorted[i-1]
        
        sum_nums2 = sum(nums2)
        initial_sum = sum(nums1)
        
        low, high = 0, n
        ans = -1
        
        while low <= high:
            mid = (low + high) // 2
            k = min(mid, n)
            reduction = prefix_nums1[k] + mid * prefix_nums2[k] - prefix_terms[k]
            total = initial_sum + mid * sum_nums2 - reduction
            if total <= x:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans if ans != -1 else -1