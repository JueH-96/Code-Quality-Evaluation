from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        counts = Counter(nums)
        distinct_nums = sorted(counts.keys())
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        mod = 10**9 + 7
        for num in distinct_nums:
            count = counts[num]
            for _ in range(count):
                for s in range(max_sum, num - 1, -1):
                    dp[s] = (dp[s] + dp[s - num]) % mod
        
        result = 0
        for s in range(l, r + 1):
            result = (result + dp[s]) % mod
        return result