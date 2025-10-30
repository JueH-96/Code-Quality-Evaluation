class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        max_len = 0
        for x in nums:
            for v in [x + 1, x]:
                prev = v - 1
                current_length = dp.get(prev, 0) + 1
                if current_length > dp.get(v, 0):
                    dp[v] = current_length
                    if current_length > max_len:
                        max_len = current_length
        return max_len if max_len != 0 else 1