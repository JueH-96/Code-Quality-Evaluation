class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}  # dp[val] = length of longest consecutive subsequence ending at val

        for num in nums:
            # Case 1: Don't increment num
            length1 = dp.get(num - 1, 0) + 1
            dp[num] = length1

            # Case 2: Increment num to num + 1
            length2 = dp.get(num, 0) + 1
            dp[num + 1] = max(dp.get(num + 1, 0), length2 -1 if num in dp else length2)

        max_len = 0
        for length in dp.values():
            max_len = max(max_len, length)

        return max_len