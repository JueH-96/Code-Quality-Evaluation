from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        points = sorted([(x, y) for x, y in zip(nums1, nums2)], key=lambda p: -p[0])
        query_indices = sorted(range(len(queries)), key=lambda i: -queries[i][0])
        result = [-1] * len(queries)
        i = 0
        mono_stack = []
        
        for qi in query_indices:
            x, y = queries[qi]
            while i < len(points) and points[i][0] >= x:
                px, py = points[i]
                while mono_stack and mono_stack[-1][1] <= py:
                    mono_stack.pop()
                if not mono_stack or mono_stack[-1][0] < px + py:
                    mono_stack.append((px + py, py))
                i += 1
            for sum_val, py in reversed(mono_stack):
                if py >= y:
                    result[qi] = sum_val
                    break
        
        return result