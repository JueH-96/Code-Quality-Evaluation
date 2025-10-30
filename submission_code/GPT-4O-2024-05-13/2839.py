from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        results = []
        
        for x_i, y_i in queries:
            max_sum = -1
            for j in range(n):
                if nums1[j] >= x_i and nums2[j] >= y_i:
                    max_sum = max(max_sum, nums1[j] + nums2[j])
            results.append(max_sum)
        
        return results