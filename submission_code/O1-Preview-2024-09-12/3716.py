class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        N = len(nums)
        max_dp = {}  # max_dp[num][d] = max length of subsequence ending with num and difference d
        result = 1
        for num in nums:
            dp_num = {}
            for d in range(0, 300):
                for delta in [d, -d]:
                    prev_num = num - delta
                    if 1 <= prev_num <= 300:
                        key = (prev_num, d)
                        prev_length = max_dp.get(key, 1)
                        length = prev_length + 1
                        dp_num[d] = max(dp_num.get(d, 2), length)
                        result = max(result, dp_num[d])
            for d in dp_num:
                key = (num, d)
                max_dp[key] = max(max_dp.get(key, 1), dp_num[d])
        return result