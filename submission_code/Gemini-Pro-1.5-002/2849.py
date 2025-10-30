class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub = sorted(nums[i:j+1])
                m = len(sub)
                cnt = 0
                for k in range(m - 1):
                    if sub[k+1] - sub[k] > 1:
                        cnt += 1
                ans += cnt
        return ans