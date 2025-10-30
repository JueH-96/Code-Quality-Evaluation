class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        nums1_to_nums2 = sorted([(nums1[i], nums2[i]) for i in range(n)], 
                                key=lambda x: -x[1])

        pq = []
        sm, res = 0, None
        for i in range(1, n + 1):
            sm += nums1_to_nums2[i - 1][1]
            heappush(pq, nums1_to_nums2[i - 1][1])
            if i > len(pq):
                heappop(pq)
            if (p := (n - i) * pq[0] + sm) <= x - sum(a for a, _ in nums1_to_nums2[i:]):
                res = min(res, i, default=i)
        return res if res is not None else -1