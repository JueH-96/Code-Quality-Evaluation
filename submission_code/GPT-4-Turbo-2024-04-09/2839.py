class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        sums = [nums1[i] + nums2[i] for i in range(n)]
        sorted_indices = sorted(range(n), key=lambda i: (nums1[i], nums2[i]))
        
        # Precompute the maximum sums that meet increasing thresholds of nums1 and nums2
        max_sums = [-1] * n
        current_max = -1
        for i in sorted_indices:
            current_max = max(current_max, sums[i])
            max_sums[i] = current_max
        
        results = []
        for x, y in queries:
            # Binary search to find the smallest index where nums1[i] >= x and nums2[i] >= y
            low, high = 0, n - 1
            valid_index = -1
            while low <= high:
                mid = (low + high) // 2
                if nums1[sorted_indices[mid]] >= x and nums2[sorted_indices[mid]] >= y:
                    valid_index = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            if valid_index == -1:
                results.append(-1)
            else:
                results.append(max_sums[sorted_indices[valid_index]])
        
        return results