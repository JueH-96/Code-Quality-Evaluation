class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        nums.sort()
        dp = [0] * (1 << m)
        cnt = [0] * (1 << m)
        for mask in range(1, 1 << m):
            for i in range(m):
                if ((mask >> i) & 1):
                    if cnt[mask ^ (1 << i)] != -1:
                        new_val = dp[mask ^ (1 << i)] + (nums[cnt[mask ^ (1 << i)]] & andValues[i])
                        if dp[mask] == 0 or new_val < dp[mask]:
                            dp[mask] = new_val
                            cnt[mask] = cnt[mask ^ (1 << i)] + 1
        return dp[-1] if dp[-1] != 0 else -1