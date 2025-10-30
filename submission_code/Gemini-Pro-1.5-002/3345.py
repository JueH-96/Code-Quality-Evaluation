class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        ans = 0
        for i in range(1 << n):
            sub = []
            for j in range(n):
                if (i >> j) & 1:
                    sub.append(nums[j])
            
            m = len(sub)
            count = 0
            for j in range(1 << m):
                s = 0
                for l in range(m):
                    if (j >> l) & 1:
                        s += sub[l]
                if s == k:
                    count += 1
            ans = (ans + count) % mod
        return ans