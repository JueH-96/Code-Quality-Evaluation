class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        max_len = 0
        for x in nums:
            for v in [x + 1, x]:
                prev = dp.get(v - 1, 0)
                current = prev + 1
                if dp.get(v, 0) < current:
                    dp[v] = current
                if current > max_len:
                    max_len = current
        return max_len