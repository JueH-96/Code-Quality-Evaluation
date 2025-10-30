class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        x -= sum(nums1)
        if x <= 0:
            return 0
        arr = list(zip(nums1, nums2))
        arr.sort(key=lambda x: x[1])
        nums1, nums2 = zip(*arr)
        nums1, nums2 = list(nums1), list(nums2)
        best = sum(nums2)
        cur = best
        for i in range(n):
            cur -= nums2[i]
            best = max(best, cur + nums1[i])
        if best <= x:
            return n
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            cur = sum(nums2[:mid])
            for i in range(mid, n):
                cur -= nums2[i]
                cur += nums1[i]
            if cur <= x:
                hi = mid
            else:
                lo = mid + 1
        return -1 if lo == n else lo