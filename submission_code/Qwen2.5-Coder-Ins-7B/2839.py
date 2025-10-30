class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        from sortedcontainers import SortedList
        n = len(nums1)
        pairs = sorted(zip(nums1, nums2), reverse=True)
        queries = sorted((x, y, i) for i, (x, y) in enumerate(queries))
        ans = [-1] * len(queries)
        sl = SortedList()
        j = 0
        for x, y, i in queries:
            while j < n and pairs[j][0] >= x:
                sl.add(pairs[j][1] + pairs[j][0])
                j += 1
            idx = sl.bisect_left(y)
            if idx < len(sl):
                ans[i] = sl[idx]
        return ans