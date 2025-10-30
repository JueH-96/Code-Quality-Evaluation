MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        cnt0 = nums.count(0)
        freq = Counter()
        for x in nums:
            if x != 0 and x <= r:
                freq[x] += 1
        
        dp = [0] * (r + 1)
        dp[0] = 1
        
        for x, c in freq.items():
            new_dp = [0] * (r + 1)
            for mod in range(x):
                s = 0
                j = 0
                while mod + j * x <= r:
                    idx = mod + j * x
                    s += dp[idx]
                    if j >= c + 1:
                        s -= dp[mod + (j - c - 1) * x]
                    s %= MOD
                    new_dp[idx] = s
                    j += 1
            dp = new_dp
        
        total = sum(dp[l:r+1]) % MOD
        result = (total * (cnt0 + 1)) % MOD
        return result