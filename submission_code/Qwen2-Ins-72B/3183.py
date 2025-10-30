class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            cnt = 0
            for num in nums:
                if num & (1 << i):
                    cnt += 1
            if cnt >= k:
                ans += (1 << i)
        return ans