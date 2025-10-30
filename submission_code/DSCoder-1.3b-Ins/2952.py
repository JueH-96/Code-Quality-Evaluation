class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        def check(mid):
            total, cnt = 0, 0
            for i in range(len(nums1)):
                total += max(0, nums1[i] - mid + nums2[i])
                if nums1[i] % 2 == 0:
                    cnt += 1
            return total <= mid and cnt <= x

        l, r = 1, 10**9
        while l < r:
            mid = (l + r) // 2
            if check(mid):
                r = mid
            else:
                l = mid + 1
        return l if check(l) else -1