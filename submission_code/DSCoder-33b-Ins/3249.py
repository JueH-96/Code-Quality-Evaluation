class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        xor = 0
        cnt = [0] * 22
        for i in range(n):
            xor ^= nums[i]
            for j in range(22):
                if (xor >> j) & 1:
                    cnt[j] += 1
        if xor == k:
            return n - max(cnt)
        if k == 0:
            return cnt[0]
        res = float('inf')
        for i in range(22):
            if (k >> i) & 1:
                res = min(res, n - cnt[i])
        return res