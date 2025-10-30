class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        for num in nums:
            len_num = dp.get(num - 1, 0) + 1
            len_num_plus_1 = dp.get(num, 0) + 1
            dp[num] = max(dp.get(num, 0), len_num)
            dp[num + 1] = max(dp.get(num + 1, 0), len_num_plus_1)
        max_len = 0
        for val in dp.values():
            max_len = max(max_len, val)
        return max_len