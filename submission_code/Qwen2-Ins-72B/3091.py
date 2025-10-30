class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = [0] * 20001
        for num in nums:
            freq[num] += 1

        dp = [1] + [0] * r
        for num, count in enumerate(freq):
            for i in range(r, num - 1, -1):
                for j in range(1, count + 1):
                    if i - num * j >= 0:
                        dp[i] += dp[i - num * j]
                    else:
                        break
        return sum(dp[l:r+1]) % MOD