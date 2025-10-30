from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into a list of tuples (nums1[i], nums2[i])
        nums = sorted([(nums1[i], nums2[i]) for i in range(len(nums1))], reverse=True)
        
        # Process queries in descending order of x_i
        queries = sorted(enumerate(queries), key=lambda x: x[1][0], reverse=True)
        
        result = [-1] * len(queries)
        active = []
        
        for q_idx, (x, y) in queries:
            # Add all pairs (nums1[i], nums2[i]) to active where nums1[i] >= x
            while nums and nums[-1][0] >= x:
                a, b = nums.pop()
                # Maintain active in decreasing order of nums2[i]
                while active and active[-1][1] <= b:
                    active.pop()
                active.append((a, b))
            
            # Find the maximum sum where nums2[i] >= y
            idx = bisect.bisect_left(active, (0, y), key=lambda x: x[1])
            if idx < len(active):
                result[q_idx] = active[idx][0] + active[idx][1]
        
        return result