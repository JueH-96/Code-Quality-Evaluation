class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        from sortedcontainers import SortedList
        pairs = sorted(zip(nums1, nums2))
        res = []
        active = SortedList()
        j = len(queries) - 1
        for x, y in sorted(((y, x, i) for i, (x, y) in enumerate(queries)), reverse=True):
            while j >= 0 and pairs[j][0] >= x:
                active.add(pairs[j][0] + pairs[j][1])
                j -= 1
            i = active.bisect_right(y)
            res.append((active[i - 1] if i else -1, i))
        return [x for x, _ in sorted(res, key=lambda x: x[1])]