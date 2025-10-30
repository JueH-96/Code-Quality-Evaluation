class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        def check(x):
            cnt = [0] * 128
            for num in nums:
                if num & x:
                    cnt[num & x] += 1
            return max(cnt)

        l, r = 1, 2**32
        while l < r:
            mid = (l + r) // 2
            if check(mid) > 0:
                l = mid + 1
            else:
                r = mid
        return l - 1