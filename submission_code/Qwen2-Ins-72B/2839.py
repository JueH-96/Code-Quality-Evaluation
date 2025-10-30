from sortedcontainers import SortedList

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        m = len(queries)
        events = []
        for i, (x, y) in enumerate(queries):
            events.append((x, y, i, 1))
        for i in range(n):
            events.append((nums1[i], nums2[i], i, 0))
        events.sort(reverse=True)
        ans = [-1] * m
        sl = SortedList()
        for x, y, idx, q in events:
            if q:
                sl.add(nums1[idx] + nums2[idx])
            else:
                if sl.bisect_right(x + y) > 0:
                    ans[queries[idx][2]] = sl[sl.bisect_right(x + y) - 1]
        return ans